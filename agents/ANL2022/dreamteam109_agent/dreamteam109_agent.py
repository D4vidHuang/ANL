import datetime
import json
import logging
import os
from math import floor
from random import randint, random
import time
from decimal import Decimal
from os import path
from typing import TypedDict, cast, List, Dict

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
from geniusweb.opponentmodel.FrequencyOpponentModel import FrequencyOpponentModel

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
                    # obtain the name of the opponent, cutting of the position ID.
                    self.other_name = str(actor).rsplit("_", 1)[0]
                    self.attempt_load_data()
                    self.learn_from_past_sessions()

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
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
                # self.opponent_model = OpponentModel(self.domain, self.logger)
                self.opponent_model = FrequencyOpponentModel.With(self, self.domain, self.profile.getReservationBid())

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            # self.opponent_model.update(bid)
            self.opponent_model.WithAction(action, self.progress)
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

        if hasattr(self, 'last_time') and self.last_time is not None:
            self.round_times.append(time.time() * 1000 - self.last_time)
            self.avg_time = sum(self.round_times[-3:]) / len(self.round_times[-3:])
        self.last_time = time.time() * 1000

        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)
            self.logger.log(logging.INFO, f"Accepting bid: {self.last_received_bid}")
        else:
            bid = self.find_bid()
            t = self.progress.get(time.time() * 1000)

            if t >= 0.95 and hasattr(self, 'opponent_bid_times') and len(self.opponent_bid_times) > 0:
                t_o = self.regression_opponent_time(self.opponent_bid_times[-10:])
                self.logger.log(logging.INFO, f"Current time: {t}, Predicted opponent response time: {t_o}")
                while t < 1 - t_o:
                    t = self.progress.get(time.time() * 1000)

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
            self.logger.log(logging.INFO, f"Loaded previous data for opponent: {self.other}")
        else:
            self.logger.log(logging.INFO, f"No previous data found for opponent: {self.other}")
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
        self.logger.log(logging.INFO, "Session data updated.")

    def save_data(self):
        if not self.other:
            self.logger.log(logging.WARNING, "Opponent name not set; skipping data save.")
            return
        data_file_path = self.get_data_file_path()
        with open(data_file_path, 'w') as file:
            json.dump(self.data_dict, file, indent=4)
        self.logger.log(logging.INFO, f"Data saved for opponent: {self.other}")

    def learn_from_past_sessions(self):
        """根据以前的会话学习并调整策略参数"""
        sessions = self.data_dict.get("sessions", [])
        if not sessions:
            return

        failed_sessions = [s for s in sessions if s["utilityAtFinish"] < 0.5]
        good_sessions = [s for s in sessions if s["utilityAtFinish"] >= 0.5]

        self.adjust_strategy_based_on_history(failed_sessions, good_sessions)
        self.logger.log(logging.INFO, "Strategy adjusted based on past sessions.")

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

        self.logger.log(logging.INFO,
                        f"Adjusted strategies: min_util={self.min_util}, top_bids_percentage={self.top_bids_percentage}, "
                        f"force_accept_at_remaining_turns={self.force_accept_at_remaining_turns}, "
                        f"force_accept_at_remaining_turns_light={self.force_accept_at_remaining_turns_light}")

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
        self.logger.log(logging.INFO, "Finding bid...")

        # 初始化或更新所有可能报价的效用列表
        if self.bids_with_utilities is None or len(self.bids_with_utilities) == 0:
            self.logger.log(logging.INFO, "Calculating bids with utilities...")
            startTime = time.time()
            self.calculate_bids_with_utilities()
            endTime = time.time()
            self.logger.log(logging.INFO, f"Calculated bids with utilities in {endTime - startTime} seconds.")

        # 谈判进展阶段判断
        progress = self.progress.get(time.time() * 1000)

        # 如果接近谈判末端并有对手的最佳报价，直接考虑使用
        if progress > 0.95 and hasattr(self, 'opponent_best_bid') and self.opponent_best_bid is not None:
            return self.opponent_best_bid

        # 根据谈判进度选择策略
        if progress < 0.5:
            # 谈判前半段随机探索
            return self.random_explore()
        else:
            # 谈判后半段选择效用值较高的报价
            return self.choose_high_utility_bid(progress)

    def calculate_bids_with_utilities(self):
        """计算所有可能报价的效用值，并排序。"""
        self.bids_with_utilities = []
        for bid in self.all_bids:
            utility = self.profile.getUtility(bid)
            self.bids_with_utilities.append((bid, utility))
        self.bids_with_utilities.sort(key=lambda x: x[1], reverse=True)

    def random_explore(self):
        """随机探索报价空间，选择一个报价。"""
        index = randint(0, len(self.bids_with_utilities) - 1)
        return self.bids_with_utilities[index][0]

    def choose_high_utility_bid(self, progress):
        """根据谈判进展选择一个高效用值的报价。"""
        # 考虑使用动态比例选择报价
        top_percentage = max(5, len(self.bids_with_utilities) * self.top_bids_percentage * (1 - progress))
        top_index = int(min(len(self.bids_with_utilities) - 1, top_percentage))
        return self.bids_with_utilities[top_index][0]

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
            # opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_utility = self.opponent_model.getUtility(bid)
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
