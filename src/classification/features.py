TARGET = "IsCanceled"

LEAKAGE_COLUMNS = [
    "ReservationStatus",
    "ReservationStatusDate",
]

LIFECYCLE_COLUMNS = [
    "AssignedRoomType",
    "BookingChanges",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

OTHER_DROPPED = [
    "ADR",
    "ArrivalDateWeekNumber",
]

DROP_COLUMNS = LEAKAGE_COLUMNS + LIFECYCLE_COLUMNS + OTHER_DROPPED