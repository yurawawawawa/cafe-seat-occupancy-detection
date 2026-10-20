import unittest
from collections import deque

import numpy as np

from occupancy import (
    AVAILABLE,
    OCCUPIED,
    UNKNOWN,
    MatchingConfig,
    PersonDetection,
    SeatRegion,
    calculate_occupancy_score,
    match_persons_to_seats,
    update_seat_history,
)


def square(seat_id: str, left: int, top: int = 0, size: int = 100, anchor=None):
    return SeatRegion(
        seat_id,
        np.array(
            [
                [left, top],
                [left + size, top],
                [left + size, top + size],
                [left, top + size],
            ],
            dtype=np.int32,
        ),
        anchor=anchor,
    )


class OccupancyMatchingTests(unittest.TestCase):
    def test_one_person_inside_one_seat(self):
        person = PersonDetection(0, (20, 20, 80, 80), 0.9)
        matches, _ = match_persons_to_seats([person], [square("seat_01", 0)])
        self.assertEqual(matches[0].seat_id, "seat_01")

    def test_one_person_overlapping_two_seats_is_assigned_once(self):
        person = PersonDetection(0, (45, 20, 105, 80), 0.9)
        seats = [square("seat_01", 0), square("seat_02", 50)]
        matches, _ = match_persons_to_seats([person], seats)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len({match.seat_id for match in matches.values()}), 1)

    def test_two_persons_are_globally_assigned_to_two_seats(self):
        persons = [
            PersonDetection(0, (10, 20, 60, 80), 0.9),
            PersonDetection(1, (110, 20, 160, 80), 0.9),
        ]
        seats = [square("seat_01", 0), square("seat_02", 100)]
        matches, _ = match_persons_to_seats(persons, seats)
        self.assertEqual({match.seat_id for match in matches.values()}, {"seat_01", "seat_02"})

    def test_standing_person_near_seat_is_not_automatically_matched(self):
        person = PersonDetection(0, (105, 10, 155, 70), 0.95)
        matches, _ = match_persons_to_seats([person], [square("seat_01", 0)])
        self.assertEqual(matches, {})

    def test_low_confidence_can_be_rejected_by_configured_threshold(self):
        person = PersonDetection(0, (20, 20, 80, 80), 0.20)
        seat = square("seat_01", 0)
        self.assertLess(
            calculate_occupancy_score(person, seat),
            calculate_occupancy_score(PersonDetection(0, person.bbox, 0.95), seat),
        )
        strict = MatchingConfig(min_match_score=0.96)
        matches, _ = match_persons_to_seats([person], [seat], strict)
        self.assertEqual(matches, {})

    def test_no_people_produces_unknown_not_available(self):
        history = {"seat_01": deque(maxlen=8)}
        last_status = {"seat_01": UNKNOWN}
        config = MatchingConfig()
        for _ in range(5):
            statuses = update_seat_history(history, last_status, {"seat_01": None}, config)
        self.assertEqual(statuses["seat_01"], UNKNOWN)

    def test_temporary_missing_detection_does_not_clear_occupied_seat(self):
        history = {"seat_01": deque(maxlen=8)}
        last_status = {"seat_01": UNKNOWN}
        config = MatchingConfig()
        for _ in range(3):
            update_seat_history(history, last_status, {"seat_01": True}, config)
        for _ in range(3):
            statuses = update_seat_history(history, last_status, {"seat_01": None}, config)
        self.assertEqual(statuses["seat_01"], OCCUPIED)

    def test_overlapping_polygons_never_receive_same_person(self):
        persons = [
            PersonDetection(0, (40, 20, 100, 80), 0.9),
            PersonDetection(1, (90, 20, 150, 80), 0.9),
        ]
        seats = [square("seat_01", 0), square("seat_02", 50)]
        matches, _ = match_persons_to_seats(persons, seats)
        seat_ids = [match.seat_id for match in matches.values()]
        self.assertEqual(len(seat_ids), len(set(seat_ids)))

    def test_available_requires_observed_negative_evidence(self):
        history = {"seat_01": deque(maxlen=8)}
        last_status = {"seat_01": UNKNOWN}
        config = MatchingConfig()
        for _ in range(4):
            statuses = update_seat_history(history, last_status, {"seat_01": False}, config)
        self.assertEqual(statuses["seat_01"], AVAILABLE)


if __name__ == "__main__":
    unittest.main()
