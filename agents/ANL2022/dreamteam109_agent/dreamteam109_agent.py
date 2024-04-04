import datetime
import json
import logging
import os
from math import floor
from random import randint, random
import time
from decimal import Decimal
from os import path
from typing import TypedDict, cast, List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
import itertools

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.issuevalue.Value import Value
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger
from .utils.logger import Logger

from .utils.opponent_model import OpponentModel
from .utils.utils import bid_to_string

from sklearn.cluster import DBSCAN

class SessionData(TypedDict):
    progressAtFinish: float
    utilityAtFinish: float
    didAccept: bool
    isGood: bool
    topBidsPercentage: float
    forceAcceptAtRemainingTurns: float

class DataDict(TypedDict):
    sessions: list[SessionData]


class DreamTeam109Agent(DefaultParty):

    def __init__(self):
        super().__init__()
        self.logger: Logger = Logger(self.getReporter(), id(self))

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: PartyId = None
        self.other_name: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.data_dict: DataDict = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.all_bids: AllBidsList = None
        self.bids_with_utilities: list[tuple[Bid, float]] = None
        self.num_of_top_bids: int = 1
        self.min_util: float = 0.9

        self.round_times: list[Decimal] = []
        self.last_time = None
        self.avg_time = 0
        self.utility_at_finish: float = 0
        self.did_accept: bool = False
        self.top_bids_percentage: float = 1 / 300
        self.force_accept_at_remaining_turns: float = 1
        self.force_accept_at_remaining_turns_light: float = 1
        self.opponent_best_bid: Bid = None
        self.logger.log(logging.INFO, "party is initialized")
        self.opponent_bids: List[Bid] = []

        self.issues: List[str] = None
        self.sorted_issue_utility: Dict[str, List[Tuple[str, Decimal]]] = None
        self.sorted_utility: List[Tuple[Tuple, float]] = {} 
        self.weights: Dict[str, Decimal] = None 

        self.all_bids_list: List[Bid] = []
        self.reservation_bid_utility = 0.8

    
    def combination(self, utility_map: Dict[str, List[Tuple[str, Decimal]]], NUM_ISSUE: int, tmp: List[Tuple[str, Decimal]], res: List[List[Tuple[str, Decimal]]]) -> None:
        if len(tmp) == NUM_ISSUE: 
            res.append(list(tmp))
        else:
            cur_idx: int = len(tmp)
            cur_issue_name: str = self.issues[cur_idx]
            vals: List[Tuple[str, Decimal]] = utility_map[cur_issue_name]
            for val in vals: 
                tmp.append(val)
                self.combination(utility_map, NUM_ISSUE, tmp, res)
                tmp.pop()




    def preprocessing(self):
        self.issues = list(self.profile.getWeights().keys())
        NUM_ISSUE: int = len(self.issues)
        self.weights = self.profile.getWeights()
        utility_map = self.profile.getUtilities()
        for k in list(utility_map.keys()):
            v = utility_map[k]
            dv = v.getUtilities()
            sdv = list(dict(sorted(dv.items(), key=lambda it: it[1])).items())
            utility_map[k] = sdv
        self.sorted_issue_utility = utility_map
        combinations: List[List[str]] = []
        self.combination(utility_map, len(self.issues), [], combinations)
        tmp = []
        sorted_utility: Dict[Tuple, float] = {}
        for comb in combinations:
            cur_utility = 0
            for i in range(len(self.issues)): 
                cur_utility += float(self.weights[self.issues[i]]) * float(comb[i][1])
                tmp.append(comb[i][0])
            sorted_utility[tuple(tmp)] = cur_utility
            tmp = []
        self.sorted_utility = list(sorted(sorted_utility.items(), key=lambda item: item[1]))

    def k_means(self, bids: List[Bid]) -> Tuple[np.ndarray, np.ndarray]:
        #print(bids)
        utilities = []
        for b in bids: utilities.append(float(self.profile.getUtility(b)))
        #print(len(utilities))
        utilities = np.array(np.reshape(utilities, newshape=(-1, 1)))
        #print(len(bids))
        #utilities: np.ndarray = np.ndarray(map(lambda b: self.profile.getUtility(b), bids))
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(utilities)
        #print("Kmeans over")
        return kmeans.labels_, list(map(lambda c: c[0], kmeans.cluster_centers_))

    def infer_opponent_bottom_line(self):
        utilities = self.get_bid_utilities(self.all_bids_list)
        if len(utilities) < 2:
            return None
        utilities_array = np.array(utilities).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(utilities_array)
        bottom_line_estimate = min(kmeans.cluster_centers_)[0]
        return bottom_line_estimate

    def get_bid_utilities(self, bids):
        return [self.profile.getUtility(bid) for bid in bids]

    def dbscan_method(self):
        utilities = np.array([float(self.profile.getUtility(bid)) for bid in self.all_bids_list]).reshape(-1, 1)


        #这个地方你可以改到底要多少个类，就这玩意和kmeans不一样的点是它有很多个类但是会决定一些类为噪声，最后只取最好的两个
        #我尝试了一下但是好像感觉取多少类这个问题没什么解决方案，就是影响不大
        dbscan = DBSCAN(eps=0.05, min_samples=5).fit(utilities)
        labels = dbscan.labels_

        unique_labels = set(labels) - {-1}

        if not unique_labels:
            return self.random_explore(), self.random_explore()

        bids_by_cluster = {label: [] for label in unique_labels}
        for bid, label in zip(self.all_bids_list, labels):
            if label in bids_by_cluster:
                bids_by_cluster[label].append(bid)

        largest_clusters = sorted(bids_by_cluster.keys(), key=lambda x: len(bids_by_cluster[x]), reverse=True)[:2]

        representative_bids = []
        for cluster in largest_clusters:
            cluster_bids = bids_by_cluster[cluster]
            average_utility = np.mean([self.profile.getUtility(bid) for bid in cluster_bids])
            representative_bid = min(cluster_bids, key=lambda bid: abs(self.profile.getUtility(bid) - average_utility))
            representative_bids.append(representative_bid)

        while len(representative_bids) < 2:
            representative_bids.append(representative_bids[0])

        return representative_bids[0], representative_bids[1]





    def bin_search(self, bid: Bid) -> Tuple:
        tuil: float = float(self.profile.getUtility(bid))
        left, right = 0, len(self.sorted_utility) - 1
        while left <= right:
            mid = (left + right) // 2
            #print(f"Left : {left} | Right : {right} | mid : {mid} | diff : {self.sorted_utility[mid][1] - tuil}")
            if self.sorted_utility[mid][1] - tuil > 1e-5: right = mid - 1
            elif 1e-5 < tuil - self.sorted_utility[mid][1]: left = mid + 1
            else: return mid
        raise Exception("Bin search fails")




    def intp_bids(self, bid_l: Bid, bid_r: Bid, ratio: float, strict=False):
        if strict:
            #print("in")
            bid_raw: Dict[str, str] = {}
            idx_l = self.bin_search(bid_l)
            idx_r = self.bin_search(bid_r)
            idx_diff = idx_r - idx_l
            offset = round(idx_diff * ratio)
            cur_config: Tuple = self.sorted_utility[idx_l + offset][0]
            for i in range(len(self.issues)): bid_raw[self.issues[i]] = cur_config[i]
            bid = Bid(bid_raw)
            return bid

        bid_raw: Dict[str, str] = {}
        rand = np.random.choice(range(0, len(self.issues)), size=3, replace=False)
        for ci, cur_issue in enumerate(self.issues):
            if ci not in rand:
                bid_raw[cur_issue] = bid_r.getValue(cur_issue)
                continue
            cur_utility = self.sorted_issue_utility[cur_issue]
            val_l, val_r = bid_l.getValue(cur_issue), bid_r.getValue(cur_issue)
            idx_l, idx_r = None, None
            for idx, (k, v) in enumerate(cur_utility):
                if k == val_l: idx_l = idx 
                if k == val_r: idx_r = idx 
            idx_diff = idx_r - idx_l
            self.logger.log(logging.INFO, f"idx_l : {idx_l} | idx_r : {idx_r} | idx_diff: {idx_diff}")
            offset = round(idx_diff * ratio)
            self.logger.log(logging.INFO, f"ratio : {ratio} | offset : {offset} | cur_utility[idx_l] : {cur_utility[idx_l]} | cur_utility[idx_r] : {cur_utility[idx_r]} | cur_utility[idx_l + offset] : {cur_utility[idx_l + offset]}")
            bid_raw[cur_issue] = cur_utility[idx_l + offset][0]
        #print("bid")
        #print(bid_raw)
        bid = Bid(bid_raw) # 他妈为什么doc都不给一个？？？ 我都不知道怎么创建Bid。
        #print("Create bid")
        return bid

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            # compose a list of all possible bids
            self.all_bids = AllBidsList(self.domain)
            #print(dir(self.profile))
            #print(self.profile.getUtilities()['issueA'].getUtilities())   
            #print(self.profile.getWeights())
            #print(self.domain)
            self.preprocessing()
            #print(self.sorted_issue_utility)
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                if self.other is None:
                    self.other = actor
                    self.other_name = str(actor).rsplit("_", 1)[0]
                    self.attempt_load_data()  # 尝试加载历史数据并学习
                    self.learn_from_past_sessions()

                self.opponent_action(action)  # 处理对手行动，更新对手模型

        elif isinstance(data, YourTurn):
            # 执行轮次动作，结合对手模型和历史数据进行决策
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            agreements = cast(Finished, data).getAgreements()
            if len(agreements.getMap()) > 0:
                agreed_bid = agreements.getMap()[self.me]
                self.logger.log(logging.INFO, "agreed_bid = " + bid_to_string(agreed_bid))
                self.utility_at_finish = float(self.profile.getUtility(agreed_bid))
            else:
                self.logger.log(logging.INFO, "no agreed bid (timeout? some agent crashed?)")
            
            self.update_data_dict()
            self.save_data()

            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "DreamTeam109 agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain, self.logger)

            bid = cast(Offer, action).getBid()  

            self.all_bids_list.append(bid)

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

            if self.opponent_best_bid is None:
                self.opponent_best_bid = bid
            elif self.profile.getUtility(bid) > self.profile.getUtility(self.opponent_best_bid):
                self.opponent_best_bid = bid

    def my_turn(self):

        """
        首先检查是否接受对手的最后一个报价。如果不接受，则尝试寻找一个新的报价。
        如果已经接近谈判的截止时间（例如，进度大于或等于95%），它会使用预测的对手响应时间来决定是否等待再次行动，
        以增加谈判成功的可能性。同时记录行动时间，以便用于计算平均行动时间，这可能对未来的决策有影响。
        最后，它通过日志记录详细说明了决策过程。
        """
        opponent_bottom_line_estimate = self.infer_opponent_bottom_line()
        if hasattr(self, 'last_time') and self.last_time is not None:
            self.round_times.append(time.time() * 1000 - self.last_time)
            self.avg_time = sum(self.round_times[-3:]) / len(self.round_times[-3:])
        self.last_time = time.time() * 1000

        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)
            self.logger.log(logging.INFO, f"Accepting bid: {self.last_received_bid}")
        else:
            bid = self.find_bid()
            self.all_bids_list.append(bid)
            t = self.progress.get(time.time() * 1000)

            if t >= 0.95 and hasattr(self, 'opponent_bid_times') and len(self.opponent_bid_times) > 0 :
                # print("opponent_bottom_line_estimate: " + opponent_bottom_line_estimate)
                # bid = self.generate_offer_based_on_estimate(opponent_bottom_line_estimate)
                t_o = self.regression_opponent_time(self.opponent_bid_times[-10:])
                self.logger.log(logging.INFO, f"Current time: {t}, Predicted opponent response time: {t_o}")
                while t < 1 - t_o:
                    t = self.progress.get(time.time() * 1000)
            # elif t >= 0.95 and hasattr(self, 'opponent_bid_times') and len(self.opponent_bid_times) > 0:
            #     t_o = self.regression_opponent_time(self.opponent_bid_times[-10:])
            #     self.logger.log(logging.INFO, f"Current time: {t}, Predicted opponent response time: {t_o}")
            #     while t < 1 - t_o:
            #         t = self.progress.get(time.time() * 1000)
            
            action = Offer(self.me, bid)
            self.logger.log(logging.INFO, f"Offering bid: {bid}")

        # Send the action
        self.send_action(action)

    def get_data_file_path(self) -> str:
        """构建并返回存储对手数据的文件路径。"""
        filename = f"{self.other}.json"
        return os.path.join(self.storage_dir, filename)

    def attempt_load_data(self):
        """尝试加载以前保存的对手数据。"""
        data_file_path = self.get_data_file_path()
        if os.path.exists(data_file_path):
            with open(data_file_path, 'r') as file:
                self.data_dict = json.load(file)
            #self.logger.log(logging.INFO, f"Loaded previous data for opponent: {self.other}")
        else:
            #self.logger.log(logging.INFO, f"No previous data found for opponent: {self.other}")
            self.data_dict = {"sessions": []}

    def update_data_dict(self):
        """在谈判结束时更新关于对手的数据字典。这个地方我现在用的都是dreamteam的数据，最后我们要换成我们自己的
        """
        session_data = {
            "progressAtFinish": self.progress.get(time.time() * 1000),
            "utilityAtFinish": self.utility_at_finish,
            "didAccept": self.did_accept,
            "isGood": self.utility_at_finish >= self.min_util,
            "topBidsPercentage": self.top_bids_percentage,
            "forceAcceptAtRemainingTurns": self.force_accept_at_remaining_turns
        }
        self.data_dict["sessions"].append(session_data)
        #self.logger.log(logging.INFO, "Session data updated.")

    def save_data(self):
        if not self.other:
            #self.logger.log(logging.WARNING, "Opponent name not set; skipping data save.")
            return
        data_file_path = self.get_data_file_path()
        with open(data_file_path, 'w') as file:
            json.dump(self.data_dict, file, indent=4)
        #self.logger.log(logging.INFO, f"Data saved for opponent: {self.other}")

    def learn_from_past_sessions(self):
        """根据以前的会话学习并调整策略参数"""
        sessions = self.data_dict.get("sessions", [])
        if not sessions:
            return

        failed_sessions = [s for s in sessions if s["utilityAtFinish"] < 0.5]
        good_sessions = [s for s in sessions if s["utilityAtFinish"] >= 0.5]

        self.adjust_strategy_based_on_history(failed_sessions, good_sessions)
        #self.logger.log(logging.INFO, "Strategy adjusted based on past sessions.")

    def adjust_strategy_based_on_history(self, failed_sessions: List[Dict], good_sessions: List[Dict]):
        """根据历史成功和失败会话调整策略参数，这个地方可以加机器学习，我现在是用gpt整了点东西"""
        num_failed = len(failed_sessions)
        num_good = len(good_sessions)
        total_sessions = num_failed + num_good

        # 调整接受报价的策略，基于历史成功率
        if total_sessions > 0:
            success_rate = num_good / total_sessions
            self.min_util = 0.75 - 0.25 * success_rate  # 假设基线最小效用值为0.75，随着成功率提高，可以适当降低这一阈值

        # 根据历史记录调整顶级报价的选择范围
        # 如果历史记录显示较多失败的谈判，可能意味着需要在顶级报价中选择更具吸引力的报价
        self.top_bids_percentage = 1 / (50 - min(20, 5 * num_failed))  # 失败次数越多，选择范围越窄

        # 调整基于时间压力的策略
        # 如果在历史谈判中频繁失败，可能需要更早地开始考虑接受报价或提出更有吸引力的报价
        if num_failed > num_good:
            self.force_accept_at_remaining_turns *= 1.1  # 增加接受报价的倾向
            self.force_accept_at_remaining_turns_light = min(1.5,
                                                             self.force_accept_at_remaining_turns_light + 0.1)  # 轻微增加接受报价的倾向
        else:
            self.force_accept_at_remaining_turns *= 0.9  # 减少接受报价的倾向
            self.force_accept_at_remaining_turns_light = max(1,
                                                             self.force_accept_at_remaining_turns_light - 0.1)  # 轻微减少接受报价的倾向

        # 根据好的会话调整，增加冒险性
        # 如果历史记录显示较多成功的谈判，可以尝试更加冒险的策略
        if num_good > num_failed:
            self.min_util -= 0.05 * (num_good - num_failed) / total_sessions  # 成功越多，可以适当降低最小效用值阈值，尝试更冒险的报价

        # 确保所有参数都在合理范围内
        self.min_util = max(0.5, min(self.min_util, 0.95))  # 保持最小效用值在0.5到0.95之间
        self.force_accept_at_remaining_turns = max(0, min(self.force_accept_at_remaining_turns, 2))  # 保持接受报价的倾向在合理范围
        self.force_accept_at_remaining_turns_light = max(0, min(self.force_accept_at_remaining_turns_light, 2))  # 同上

        #self.logger.log(logging.INFO,
        #                f"Adjusted strategies: min_util={self.min_util}, top_bids_percentage={self.top_bids_percentage}, "
        #                f"force_accept_at_remaining_turns={self.force_accept_at_remaining_turns}, "
        #                f"force_accept_at_remaining_turns_light={self.force_accept_at_remaining_turns_light}")

    def did_fail(self, session: SessionData):
        return session["utilityAtFinish"] == 0

    def low_utility(self, session: SessionData):
        return session["utilityAtFinish"] < 0.5

    def accept_condition(self, bid: Bid) -> bool:
        """
        根据谈判的平均时间和剩余时间动态调整接受报价的条件
        报价效用高于最小效用阈值或在接近截止时间时报价好于平均水平的报价都可以被接受，能够适应多变的谈判情境。
        考虑了历史上的报价效用分布来判断当前报价是否具有竞争力。
        """
        if bid is None:
            return False

        # 获取谈判的当前进度
        progress = self.progress.get(time.time() * 1000)
        if progress < 0.95: return False

        # 动态调整接受报价的阈值，考虑历史数据和谈判平均时间
        dynamic_threshold = self.calculate_dynamic_threshold(progress)

        # 采用一个更加灵活的策略，考虑报价的效用、谈判的进展以及对手的可能行为
        is_high_utility = self.profile.getUtility(bid) >= self.min_util
        is_close_to_deadline = progress >= dynamic_threshold
        is_better_than_average_offers = self.is_bid_better_than_average(bid, progress)

        # 如果报价的效用高于最小效用阈值，或者谈判接近截止且报价好于平均报价，则接受
        return is_high_utility or is_close_to_deadline and is_better_than_average_offers

    def calculate_dynamic_threshold(self, progress):
        """根据谈判进展动态计算阈值"""
        # 这里的1-1000和1-5000是乱写的
        threshold = 1 - 1000 * self.force_accept_at_remaining_turns * self.avg_time / self.progress.getDuration()
        light_threshold = 1 - 5000 * self.force_accept_at_remaining_turns_light * self.avg_time / self.progress.getDuration()
        return max(0.95, min(threshold, light_threshold))  # 确保阈值在一个合理的范围内

    def is_bid_better_than_average(self, bid, progress):
        """判断当前报价是否好于过去接收的平均报价"""
        if not hasattr(self, 'bids_with_utilities') or not self.bids_with_utilities:
            return False  # 如果没有之前的报价作为参考，直接返回False

        # 根据进度动态选择比较的报价范围
        if progress < 0.5:
            compare_index = len(self.bids_with_utilities) // 2  # 前半段谈判比较中等报价
        else:
            compare_index = len(self.bids_with_utilities) // 5  # 谈判后期比较顶部20%的报价

        average_utility = self.bids_with_utilities[compare_index][1]
        return self.profile.getUtility(bid) >= average_utility

    """
        一次性计算效用：在需要时计算并排序所有报价的效用值，避免重复计算。
        谈判进展动态调整：根据谈判的进展，选择不同的策略。在谈判的早期阶段，采用随机探索以发现潜在的高效用报价；在谈判的后期，则优先考虑效用值较高的报价，以提高成功的可能性。
        考虑对手最佳报价：如果接近谈判末端且存在对手的最佳报价，直接考虑使用该报价，以提高达成协议的机会。
        """

    def find_bid(self) -> Bid:
        #self.logger.log(logging.INFO, "Finding bid...")

        # 初始化或更新所有可能报价的效用列表
        if self.bids_with_utilities is None or len(self.bids_with_utilities) == 0:
            #self.logger.log(logging.INFO, "Calculating bids with utilities...")
            startTime = time.time()
            self.calculate_bids_with_utilities()
            endTime = time.time()
            #self.logger.log(logging.INFO, f"Calculated bids with utilities in {endTime - startTime} seconds.")

        # 谈判进展阶段判断
        progress = self.progress.get(time.time() * 1000)

        # 如果接近谈判末端并有对手的最佳报价，直接考虑使用
        #if progress > 0.95 and hasattr(self, 'opponent_best_bid') and self.opponent_best_bid is not None:
            #return self.opponent_best_bid

        # 根据谈判进度选择策略
        if progress < 0.1:
            #print("Progress 1")
            # 谈判前半段随机探索
            return self.random_explore()
        elif progress <= 1:
            #print("Progress 2")
            try:
                return self.intp_method()
            except Exception as e:
                quit()
        else:
            # 谈判后半段选择效用值较高的报价
            return self.choose_high_utility_bid(progress)
        
    def find_bid_match_centres(self, bids: List[Bid], labels: np.ndarray, centres: np.ndarray):
        best_bids = [None, None]
        dsts = [1, 1]
        for idx, bid in enumerate(bids):
            cur_utility = float(self.profile.getUtility(bid))
            cur_label = labels[idx]
            if best_bids[cur_label] is None or abs(cur_utility - centres[cur_label]) < dsts[cur_label]:
                best_bids[cur_label] = bid 
                #print("Before")
                #print(cur_utility, centres[cur_label])
                #print(type(cur_utility), type(centres[cur_label]))
                #print(cur_utility - centres[cur_label])
                #print("Good")
                dsts[cur_label] = abs(cur_utility - centres[cur_label])
        return best_bids

    def get_ratio(self, cond=True): 
        fixed: float = 0.2 * (np.cos(self.progress.get(time.time() * 1000) * np.pi) + 1) * 0.5
        flt: float = 0.2 * np.random.random()
        self.logger.log(logging.INFO, f"Fixed : {fixed} | Float : {flt}")
        ratio = 0.6 + fixed + flt
        return ratio

        
    def intp_method(self):
        #print("INTP Method is used.")
        #labels, centers = self.k_means(self.all_bids_list)
        #bid_0, bid_1 = self.find_bid_match_centres(self.all_bids_list, labels, centers)
        bid_0, bid_1 = self.dbscan_method()
        bid_l = bid_0 if self.profile.getUtility(bid_0) < self.profile.getUtility(bid_1) else bid_1
        bid_r = bid_0 if self.profile.getUtility(bid_0) >= self.profile.getUtility(bid_1) else bid_1
        #self.logger.log(logging.INFO, str(bid_l) + " " + str(self.profile.getUtility(bid_l)))
        #self.logger.log(logging.INFO, str(bid_r) + " " + str(self.profile.getUtility(bid_r)))
        cur_ratio = self.get_ratio()
        self.logger.log(logging.INFO, f"[cur_ratio] cur_ratio : {cur_ratio}")
        #print("Ratio")
        intp = self.intp_bids(bid_l, bid_r, cur_ratio, strict=True)
        self.logger.log(logging.INFO, "INTP Result : " + str(intp) + " " + str(self.profile.getUtility(intp)))
        return intp



    def calculate_bids_with_utilities(self):
        """计算所有可能报价的效用值，并排序。"""
        self.bids_with_utilities = []
        for bid in self.all_bids:
            utility = self.profile.getUtility(bid)
            self.bids_with_utilities.append((bid, utility))
        self.bids_with_utilities.sort(key=lambda x: x[1], reverse=True)

    def random_explore(self):

        reservation_bid_utility = 0.9

        index = randint(0, len(self.bids_with_utilities) - 1)
        chosen_bid_utility = self.bids_with_utilities[index][1]

        if chosen_bid_utility < reservation_bid_utility:
            mapped_utility = reservation_bid_utility + (1 - reservation_bid_utility) * float(chosen_bid_utility)
            closest_bid = min(self.bids_with_utilities, key=lambda x: abs(float(x[1]) - mapped_utility))
            return closest_bid[0]

        return self.bids_with_utilities[index][0]

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid with dynamic adjustments and stochastic elements.

        Args:
            bid (Bid): Bid to score.
            alpha (float, optional): Base trade-off factor between self-interested and altruistic behavior.
            eps (float, optional): Base time pressure factor.

        Returns:
            float: Adjusted score of the bid.
        """
        if bid is None:
            return 0

        progress = self.progress.get(time.time() * 1000)

        # Dynamic adjustment based on negotiation progress
        dynamic_alpha, dynamic_eps = self.dynamic_adjustments(alpha, eps, progress)

        # Calculate time pressure
        time_pressure = 1.0 - progress ** (1 / dynamic_eps)
        utility = float(self.profile.getUtility(bid))

        # Initial score based on our utility
        score = dynamic_alpha * time_pressure * utility

        # Adjust score based on opponent model, if available
        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - dynamic_alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score

    def dynamic_adjustments(self, alpha, eps, progress):
        """Dynamically adjust alpha and eps based on negotiation progress and stochastic elements."""
        # Introduce randomness in adjustment
        stochastic_factor = random.uniform(-0.05, 0.05)

        # Adjust alpha based on negotiation progress and add stochastic factor
        adjusted_alpha = alpha + (0.05 if progress > 0.8 else -0.05) + stochastic_factor

        # Ensure adjusted alpha remains within reasonable bounds
        adjusted_alpha = max(0.8, min(1.0, adjusted_alpha))

        # Adjust eps to increase time pressure as negotiation progresses
        adjusted_eps = eps - (progress * 0.05) + stochastic_factor

        # Ensure adjusted eps remains within reasonable bounds
        adjusted_eps = max(0.01, min(0.2, adjusted_eps))

        return adjusted_alpha, adjusted_eps
