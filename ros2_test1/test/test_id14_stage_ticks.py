from ros2_test1 import target_vision


def test_id14_stage_ticks_match_formal_contract():
    assert target_vision.TASK1_ID14_RETRACT_TICK == 100
    assert target_vision.TASK1_ID14_YELLOW_TICK == 180
    assert target_vision.TASK1_ID14_FIELD_TICK == 500
    assert target_vision.SPLITTER_RETRACT_TICK == 100
    assert target_vision.DISC_CATCH_PREP_SPLITTER_TICK == 100
    assert target_vision.DISC_CATCH_SPLITTER_READY_TICK == 100
    assert target_vision.DISC_CATCH_SPLITTER_RESET_TICK == 100
    assert target_vision.DISC_CATCH_SPLITTER_YELLOW_TICK == 180
    assert target_vision.DISC_CATCH_SPLITTER_FIELD_TICK == 500
