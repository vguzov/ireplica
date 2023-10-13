import numpy as np
from loguru import logger
from typing import List, Dict
from .utils import HAND_NAMES2INDS, HAND_INDS2NAMES, find_closest_before


class InteractionInterval:
    def __init__(self, start_time, end_time, interacting_hands_labels):
        self.start_time = start_time
        self.end_time = end_time
        self.interacting_hands_labels = interacting_hands_labels

    @property
    def interacting_hands_inds(self):
        return [HAND_NAMES2INDS[hname] for hname in self.interacting_hands_labels]

    @property
    def is_noop(self) -> bool:
        return len(self.interacting_hands_labels) == 0


class InteractionTimeline:
    @classmethod
    def from_gt_contacts(cls, contacts):
        self = cls()
        self._interaction_intervals_list = [InteractionInterval(c["start"], c["end"], c["parts"]) for c in contacts]
        return self

    @classmethod
    def from_interaction_intervals(cls, intervals: List[InteractionInterval]):
        self = cls()
        self._interaction_intervals_list = intervals
        return self

    @classmethod
    def from_predicted_contacts(cls, sequence_interval: np.ndarray, contact_preds, contact_ts, interaction_type, ignore_edge_intervals=True):
        self = cls()
        self._interaction_intervals_list = []
        self._interaction_intervals_timesplits = []
        self._interaction_intervals_framesplits = []  # offset by sequence_interval_frames[0]
        self.ignore_edge_intervals = ignore_edge_intervals
        self.interaction_type = interaction_type
        self.sequence_interval = sequence_interval
        self.sequence_interval_frames = np.asarray((find_closest_before(contact_ts, sequence_interval[0]),
                                                    find_closest_before(contact_ts, sequence_interval[1])))
        cropped_contact_preds = contact_preds[self.sequence_interval_frames[0]:self.sequence_interval_frames[1]]
        cropped_contact_ts = contact_ts[self.sequence_interval_frames[0]:self.sequence_interval_frames[1]]
        self._compute_interaction_intervals(cropped_contact_preds, cropped_contact_ts, sync_intervals=(interaction_type == "two_handed"))
        return self

    def _populate_interaction_intervals(self, interaction_intervals_framesplits, contact_preds, contact_ts):
        prev_framesplit = -1
        for framesplit in interaction_intervals_framesplits:
            if prev_framesplit >= 0:
                start_time = contact_ts[prev_framesplit]
                end_time = contact_ts[framesplit]
                interacting_hands_labels = []
                for hand_ind in sorted(HAND_INDS2NAMES.keys()):
                    if contact_preds[prev_framesplit, hand_ind]:
                        interacting_hands_labels.append(HAND_INDS2NAMES[hand_ind])
                self._interaction_intervals_list.append(InteractionInterval(start_time, end_time, interacting_hands_labels))
            prev_framesplit = framesplit

    def _compute_interaction_intervals(self, contact_preds, contact_ts, sync_intervals=False):
        if sync_intervals:
            logger.debug("Syncing intervals")
            contact_preds = np.minimum(*contact_preds.T)
            contact_preds = np.stack([contact_preds, contact_preds], axis=1)
            # cinds = np.nonzero(sm_contact_preds[sequence_interval_frames[0]:sequence_interval_frames[1]])[0]
            # if len(cinds) > 0:
            #     new_contacts_int = [contact_ts[sequence_interval_frames[0]:sequence_interval_frames[1]][cinds[0]],
            #                         contact_ts[sequence_interval_frames[0]:sequence_interval_frames[1]][cinds[-1]]]
            #     new_contacts_int = [new_contacts_int, new_contacts_int]
            # else:
            #     new_contacts_int = [None, None]

        # else:
        #     arr_cinds = np.nonzero(contact_preds[sequence_interval_frames[0]:sequence_interval_frames[1]])
        #     cinds = [arr_cinds[0][arr_cinds[1] == hand_ind] for hand_ind in range(2)]
        #     new_contacts_int = [([contact_ts[sequence_interval_frames[0]:sequence_interval_frames[1]][cinds[hand_ind][0]],
        #                           contact_ts[sequence_interval_frames[0]:sequence_interval_frames[1]][cinds[hand_ind][-1]]] if len(
        #         cinds[hand_ind]) > 0 else None) for hand_ind in range(2)]
        logger.debug("Detecting splits")
        diff_mask = contact_preds[:-1, :] != contact_preds[1:, :]

        self._interaction_intervals_framesplits = np.unique(np.nonzero(diff_mask)[0]) + 1  # already sorted

        self._interaction_intervals_timesplits = contact_ts[self._interaction_intervals_framesplits]

        if len(self._interaction_intervals_framesplits) == 0:
            return

        if not self.ignore_edge_intervals:
            interaction_intervals_framesplits = np.concatenate([np.zeros(1, dtype=int), self._interaction_intervals_framesplits,
                                                                np.full(1, len(self._interaction_intervals_framesplits), dtype=int)], axis=0)
        else:
            interaction_intervals_framesplits = self._interaction_intervals_framesplits
        logger.debug("Populating intervals")
        self._populate_interaction_intervals(interaction_intervals_framesplits, contact_preds, contact_ts)

    def __len__(self) -> int:
        return len(self._interaction_intervals_list)

    def __getitem__(self, ind: int) -> InteractionInterval:
        return self._interaction_intervals_list[ind]

    def __iter__(self):
        return iter(self._interaction_intervals_list)


class MultiobjectInteractionTimeline:
    def __init__(self, int_timelines: Dict[str, InteractionTimeline]):
        self.int_timelines = int_timelines

    def find_closest_active_interval_after(self, timestamp, ignore_set=set()):
        intervals_inds = {}
        overall_closest_interval = None
        overall_closest_interval_name = None
        for name, int_timeline in self.int_timelines.items():
            closest_ind = 0
            while closest_ind < len(int_timeline) and (
                    int_timeline[closest_ind].start_time < timestamp or int_timeline[closest_ind].is_noop or int_timeline[closest_ind] in ignore_set):
                closest_ind += 1
            if closest_ind < len(int_timeline):
                intervals_inds[name] = closest_ind
                if overall_closest_interval is None or overall_closest_interval.start_time > int_timeline[closest_ind].start_time:
                    overall_closest_interval = int_timeline[closest_ind]
                    overall_closest_interval_name = name
        return overall_closest_interval, overall_closest_interval_name
