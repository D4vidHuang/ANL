import json
import logging
import os

from random import randint, random
import time
from decimal import Decimal

from typing import TypedDict, cast, List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans

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

from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters

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


class GroupAgent(DefaultParty):

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

    def combination(self, utility_map: Dict[str, List[Tuple[str, Decimal]]], NUM_ISSUE: int,
                    tmp: List[Tuple[str, Decimal]], res: List[List[Tuple[str, Decimal]]]) -> None:
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
        """
            Performs K-means clustering on the utilities of given bids.

            This function takes a list of Bid objects and performs K-means clustering
            to categorize them into 2 clusters based on their utility values. It returns
            the labels for each bid indicating its cluster, and the cluster centers.

            Parameters:
            - bids: List[Bid]
                A list of Bid objects. A Bid object has some attributes
                and methods, including a method to calculate its utility.

            Returns:
            - Tuple[np.ndarray, np.ndarray]
                The first ndarray contains the labels for each bid, indicating which
                cluster the bid belongs to (0 or 1). The second ndarray is a list of the
                cluster centers' utility values.
            """

        utilities = []
        for b in bids: utilities.append(float(self.profile.getUtility(b)))

        utilities = np.array(np.reshape(utilities, newshape=(-1, 1)))

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(utilities)

        return kmeans.labels_, list(map(lambda c: c[0], kmeans.cluster_centers_))

    def infer_opponent_bottom_line(self):
        """
            Estimates the opponent's bottom line in a negotiation scenario.

            This method applies K-means clustering to the utility values of all possible bids,
            assuming that these utilities represent how the opponent values each bid. The
            opponent's bottom line is estimated as the lower center of the two clusters
            formed by the K-means algorithm, which represents the lower bound of utility
            values that the opponent might be willing to accept.

            The method returns the estimated utility value of the opponent's bottom line.
            If there are fewer than two utility values available (meaning clustering cannot
            be applied), the method returns None, indicating an inability to estimate the
            bottom line.

            Returns:
                bottom_line_estimate (float or None): The estimated utility value of the
                opponent's bottom line or None if the bottom line cannot be estimated.
            """

        utilities = self.get_bid_utilities(self.all_bids_list)
        if len(utilities) < 2:
            return None
        utilities_array = np.array(utilities).reshape(-1, 1)
        # Estimate the bottom line as the minimum of the two cluster centers.
        # This is based on the assumption that lower utility values represent
        # less favorable outcomes for the opponent.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(utilities_array)
        bottom_line_estimate = min(kmeans.cluster_centers_)[0]
        return bottom_line_estimate

    def get_bid_utilities(self, bids):
        return [self.profile.getUtility(bid) for bid in bids]

    def dbscan_method(self):
        """
            Uses DBSCAN clustering to find representative bids from clusters of bid utilities.

            This method processes the utilities of all bids in `self.all_bids_list` using
            the DBSCAN algorithm to form clusters based on utility values. It aims to identify
            two representative bids, ideally from the two largest clusters. If DBSCAN doesn't
            find any clusters, it resorts to exploring randomly.

            Returns:
                Tuple[Bid, Bid]: A tuple containing two bids that are considered representative
                of the largest clusters found by DBSCAN. If only one cluster is found or if
                no clusters are identified, it returns two identical bids or two random bids.
            """
        utilities = np.array([float(self.profile.getUtility(bid)) for bid in self.all_bids_list]).reshape(-1, 1)


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
        """
           Performs binary search to find the index of a bid based on its utility value.

           This method searches for the given bid's utility value within a sorted list
           of utility values (`self.sorted_utility`) using binary search. The list is
           expected to be sorted by utility values in ascending order. The search
           considers a small tolerance (1e-5) to account for floating-point arithmetic
           precision.

           Parameters:
           - bid: Bid
               The bid for which to find the corresponding utility value index in the
               sorted utilities list.

           Returns:
           - int: The index of the utility value in `self.sorted_utility` that matches
             the utility of the given bid, considering the defined tolerance.

           Raises:
           - Exception: If the binary search fails to find a matching utility value
             within the tolerance.
           """

        tuil: float = float(self.profile.getUtility(bid))
        left, right = 0, len(self.sorted_utility) - 1
        while left <= right:
            mid = (left + right) // 2
            # print(f"Left : {left} | Right : {right} | mid : {mid} | diff : {self.sorted_utility[mid][1] - tuil}")
            if self.sorted_utility[mid][1] - tuil > 1e-5:
                right = mid - 1
            elif 1e-5 < tuil - self.sorted_utility[mid][1]:
                left = mid + 1
            else:
                return mid
        raise Exception("Bin search fails")

    def intp_bids(self, bid_l: Bid, bid_r: Bid, ratio: float, strict=False):
        """
            Interpolates between two bids based on a given ratio, optionally using a strict method.

            Parameters:
            - bid_l: Bid
                The left (starting) bid in the interpolation.
            - bid_r: Bid
                The right (ending) bid in the interpolation.
            - ratio: float
                The ratio at which to interpolate between the two bids.
            - strict: bool, optional
                If True, the interpolation is strict, meaning it directly calculates the
                offset based on the sorted utility list. If False, it interpolates based on
                randomly selected issues and their utility values, default is False.

            Returns:
            - Bid: A new bid interpolated between bid_l and bid_r according to the specified ratio
                   and method (strict or not).

            Detailed information can be found in the report.
            """
        if strict:

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
            self.logger.log(logging.INFO,
                            f"ratio : {ratio} | offset : {offset} | cur_utility[idx_l] : {cur_utility[idx_l]} | cur_utility[idx_r] : {cur_utility[idx_r]} | cur_utility[idx_l + offset] : {cur_utility[idx_l + offset]}")
            bid_raw[cur_issue] = cur_utility[idx_l + offset][0]

        bid = Bid(bid_raw)

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


            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")


            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()

            self.all_bids = AllBidsList(self.domain)

            self.preprocessing()

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
                    self.attempt_load_data()
                    self.learn_from_past_sessions()

                self.opponent_action(action)

        elif isinstance(data, YourTurn):

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
        return "Group86 agent for the ANL 2022 competition"

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
         First check whether to accept the opponent's last offer. If not accepted, try to find a new offer.
         If it is close to the negotiation deadline (for example, progress is greater than or equal to 95%), it uses the predicted opponent response time to decide whether to wait to act again,
         to increase the likelihood of successful negotiations. Action times are also recorded so that they can be used to calculate average action times, which may have implications for future decisions.
         Finally, it details the decision-making process through logging.
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
            self.all_bids_list.append(bid)
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

        filename = f"{self.other}.json"
        return os.path.join(self.storage_dir, filename)

    def attempt_load_data(self):

        data_file_path = self.get_data_file_path()
        if os.path.exists(data_file_path):
            with open(data_file_path, 'r') as file:
                self.data_dict = json.load(file)

        else:

            self.data_dict = {"sessions": []}

    def update_data_dict(self):
        session_data = {
            "progressAtFinish": self.progress.get(time.time() * 1000),
            "utilityAtFinish": self.utility_at_finish,
            "didAccept": self.did_accept,
            "isGood": self.utility_at_finish >= self.min_util,
            "topBidsPercentage": self.top_bids_percentage,
            "forceAcceptAtRemainingTurns": self.force_accept_at_remaining_turns
        }
        self.data_dict["sessions"].append(session_data)

    def save_data(self):
        if not self.other:
            return
        data_file_path = self.get_data_file_path()
        with open(data_file_path, 'w') as file:
            json.dump(self.data_dict, file, indent=4)

    def learn_from_past_sessions(self):
        sessions = self.data_dict.get("sessions", [])
        if not sessions:
            return

        failed_sessions = [s for s in sessions if s["utilityAtFinish"] < 0.5]
        good_sessions = [s for s in sessions if s["utilityAtFinish"] >= 0.5]

        self.adjust_strategy_based_on_history(failed_sessions, good_sessions)


    def adjust_strategy_based_on_history(self, failed_sessions: List[Dict], good_sessions: List[Dict]):
        num_failed = len(failed_sessions)
        num_good = len(good_sessions)
        total_sessions = num_failed + num_good

        if total_sessions > 0:
            success_rate = num_good / total_sessions
            self.min_util = 0.75 - 0.25 * success_rate


        self.top_bids_percentage = 1 / (50 - min(20, 5 * num_failed))

        if num_failed > num_good:
            self.force_accept_at_remaining_turns *= 1.1
            self.force_accept_at_remaining_turns_light = min(1.5,
                                                             self.force_accept_at_remaining_turns_light + 0.1)
        else:
            self.force_accept_at_remaining_turns *= 0.9
            self.force_accept_at_remaining_turns_light = max(1,
                                                             self.force_accept_at_remaining_turns_light - 0.1)

        if num_good > num_failed:
            self.min_util -= 0.05 * (num_good - num_failed) / total_sessions

        self.min_util = max(0.5, min(self.min_util, 0.95))
        self.force_accept_at_remaining_turns = max(0, min(self.force_accept_at_remaining_turns, 2))
        self.force_accept_at_remaining_turns_light = max(0, min(self.force_accept_at_remaining_turns_light, 2))

    def did_fail(self, session: SessionData):
        return session["utilityAtFinish"] == 0

    def low_utility(self, session: SessionData):
        return session["utilityAtFinish"] < 0.5

    def accept_condition(self, bid: Bid) -> bool:
        """
         Dynamically adjust the conditions for accepting an offer based on average time and remaining time for negotiations
         Offers whose utility is higher than the minimum utility threshold or offers that are better than the average near the deadline can be accepted and can adapt to changing negotiation situations.
         The historical offer utility distribution is considered to determine whether the current offer is competitive.
         """
        if bid is None:
            return False

        progress = self.progress.get(time.time() * 1000)
        if progress < 0.95: return False

        dynamic_threshold = self.calculate_dynamic_threshold(progress)

        is_high_utility = self.profile.getUtility(bid) >= self.min_util
        is_close_to_deadline = progress >= dynamic_threshold
        is_better_than_average_offers = self.is_bid_better_than_average(bid, progress)

        return is_high_utility or is_close_to_deadline and is_better_than_average_offers

    def calculate_dynamic_threshold(self, progress):

        threshold = 1 - 1000 * self.force_accept_at_remaining_turns * self.avg_time / self.progress.getDuration()
        light_threshold = 1 - 5000 * self.force_accept_at_remaining_turns_light * self.avg_time / self.progress.getDuration()
        return max(0.95, min(threshold, light_threshold))

    def is_bid_better_than_average(self, bid, progress):
        if not hasattr(self, 'bids_with_utilities') or not self.bids_with_utilities:
            return False

        if progress < 0.5:
            compare_index = len(self.bids_with_utilities) // 2
        else:
            compare_index = len(self.bids_with_utilities) // 5

        average_utility = self.bids_with_utilities[compare_index][1]
        return self.profile.getUtility(bid) >= average_utility

    """
         Calculate utility in one go: Calculate and sort utility values for all quotes when needed, avoiding double calculations.
         Dynamic adjustment of negotiation progress: Choose different strategies according to the progress of negotiation. In the early stages of negotiation, random exploration is used to discover potential high-utility offers; in the later stages of negotiation, offers with higher utility values are given priority to increase the probability of success.
         Consider your opponent's best offer: If you are nearing the end of a negotiation and there is an opponent's best offer, consider using that offer directly to increase your chances of reaching an agreement.
    """

    def find_bid(self) -> Bid:
        if self.bids_with_utilities is None or len(self.bids_with_utilities) == 0:

            self.calculate_bids_with_utilities()

        progress = self.progress.get(time.time() * 1000)

        #Choose bidding strategy according to progress
        if progress < 0.1:
            #Random explore for the first half
            return self.random_explore()
        elif progress <= 1:

            try:
                return self.intp_method()
            except Exception as e:
                quit()

    def find_bid_match_centres(self, bids: List[Bid], labels: np.ndarray, centres: np.ndarray):
        best_bids = [None, None]
        dsts = [1, 1]
        for idx, bid in enumerate(bids):
            cur_utility = float(self.profile.getUtility(bid))
            cur_label = labels[idx]
            if best_bids[cur_label] is None or abs(cur_utility - centres[cur_label]) < dsts[cur_label]:
                best_bids[cur_label] = bid
                dsts[cur_label] = abs(cur_utility - centres[cur_label])
        return best_bids

    def get_ratio(self, cond=True):
        fixed: float = 0.2 * (np.cos(self.progress.get(time.time() * 1000) * np.pi) + 1) * 0.5
        flt: float = 0.2 * np.random.random()
        self.logger.log(logging.INFO, f"Fixed : {fixed} | Float : {flt}")
        ratio = 0.6 + fixed + flt
        return ratio

    def intp_method(self):
        """
        Interpolates a new bid between two representative bids identified by the DBSCAN method.

        This method first identifies two representative bids using the `dbscan_method`. It then
        determines which of these bids has the lower and higher utility, respectively, assigning
        them as left (bid_l) and right (bid_r) bounds for interpolation. The interpolation ratio
        is determined by `get_ratio()`, and the method interpolates between bid_l and bid_r using
        the `intp_bids` method with a strict interpolation strategy.

        Returns:
            The interpolated bid based on the current ratio and the utilities of the two
            representative bids.
        """
        bid_0, bid_1 = self.dbscan_method()
        bid_l = bid_0 if self.profile.getUtility(bid_0) < self.profile.getUtility(bid_1) else bid_1
        bid_r = bid_0 if self.profile.getUtility(bid_0) >= self.profile.getUtility(bid_1) else bid_1

        cur_ratio = self.get_ratio()
        self.logger.log(logging.INFO, f"[cur_ratio] cur_ratio : {cur_ratio}")

        intp = self.intp_bids(bid_l, bid_r, cur_ratio, strict=True)
        self.logger.log(logging.INFO, "INTP Result : " + str(intp) + " " + str(self.profile.getUtility(intp)))
        return intp

    def calculate_bids_with_utilities(self):

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

    def regression_opponent_time(self, param):
        pass
