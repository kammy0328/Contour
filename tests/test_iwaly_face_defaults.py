"""1.26.2: the face settings the 2611 IWALY batch was reviewed with are the app's own defaults (user 11/26: "배치 전용 설정이 적용 안된게
또 있을까?"): before, they were set by the batch's own configuration only, so the app retouched without them."""
from contour.retouch import engine as E


def test_reviewed_face_settings_are_defaults():
    assert E.NOSE_SIDES is True and E.NOSE_SIDES_WING == 0.35
    assert E.NASO_DEEP_NOSE == 1.0 and E.NASO_REACH == 0.12 and E.NASO_BULGE == 0.16 and E.NASO_WIDE_MUL == 1.25
    assert (E.NASO_WING_TOP, E.NASO_WING_TOP_FROM, E.NASO_WING_TOP_TO) == (1.0, -0.01, 0.03)
    assert E.NASO_NOSTRIL_HOLD == 0.03
    assert E.NASO_FOLD["broad"] == 2.0 and E.NASO_FOLD["level"] == 2.0 and tuple(E.NASO_FOLD["cap"]) == (0.12, 0.004)
    assert E.FOLD_BASE == {"nasolabial": 0.45, "mouth_lines": 0.35}
    assert tuple(E.FOLD_TOTAL_CAP) == (0.045, 0.03) and tuple(E.FOLD_YAW) == (0.7, 0.85) and E.FOLD_ZONE_OLD is False
    assert E.FOLD_COVER_GUARD == 1.0 and E.FOLD_PARSE_GUARD == 1.0 and E.HAND_GUARD == 1.0
    assert E.MOUTH_OPEN_KEEP == 0.75 and E.MOLE_KEEP == 1.0 and tuple(E.HOLD_MOLE) == (0.15, 0.25)
    assert E.FINAL_NOSE == 1.0 and E.FINAL_MOLE == 1.0 and E.FINAL_NASO_SKIN == 1.0 and E.FINAL_BUMP_LIPS == 0.1
    assert (E.CHIN_LO, E.CHIN_HI) == (0.02, 0.045)
    assert E.DARK_CHROMA == 0.3 and E.SKIN_BROW_HOLD == 0.02
    assert tuple(E.SKIN_FS_G) == (1.35, 0.7, 0.3, 0.2, 0.2, 0.45)


def test_strength_stays_with_the_sliders():
    # the batch's fixed strengths (FINAL_BUMP, FINAL_CHIN, FINAL_NASO, CHIN_DIMPLE) are not defaults: 요철, 턱주름 and 팔자주름 drive them
    assert E.FINAL_BUMP == 0.0 and E.FINAL_CHIN == 0.0 and E.FINAL_NASO == 0.0 and E.CHIN_DIMPLE == 0.0
    # a person out of focus is not dropped silently in the app (the batch's FOCUS_GATE was one job's choice)
    assert E.FOCUS_GATE is None


def test_final_pass_runs_for_the_nasolabial_slider():
    import inspect
    src = inspect.getsource(E._final_pass)
    assert '"nasolabial"' in src.split("if not (")[0]
    assert 'kfn_ = max(float(FINAL_NASO), float(np.clip(job.v.get("nasolabial", 0.0), 0, 1)))' in src


def test_the_mac_batchs_face_settings_are_defaults():
    # (the Mac IWALY batch's reviewed face settings, missed by the first comparison: user 11/26 "배치 전용 설정이 적용 안된게 또 있을까?";
    # the LNGSHOT QA at 4K showed what they guard against)
    from contour.retouch import skin as SK
    assert E.SKIN_EDGE_KEEP == 0.04 and E.FACE_HAIR_HOLD == 1.0 and E.FACE_NOSE_ADD == 0.8
    assert E.FOLD_EDGE_FADE == 1.0 and E.FINAL_NASO_IN_BAND == 1.0 and E.FOLD_LEVEL_LIFT_ONLY is True
    assert tuple(SK.FS_EDGE_GRAD) == (0.03, 0.08)
    assert E.CHIN_KEEP_LIP > 0 and E.CHIN_KEEP_EDGE > 0 and E.FINAL_BUMP_EDGE > 0 and E.FACE_GAP_BRIDGE == 2
