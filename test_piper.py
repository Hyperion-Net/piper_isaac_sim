"""Smoke tests for Piper arm MuJoCo setup and IK solver."""

import os
import numpy as np
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "mjcf", "piper.xml")

HOME = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0, 0.015, -0.015])

IK_STEPS = 200
DAMPING = 1e-4
MAX_DPOS = 0.05


def load_model():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    return model, data


def reset_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    for i in range(8):
        data.ctrl[i] = HOME[i]
    mujoco.mj_forward(model, data)


def solve_ik(model, data, target_pos, ee_site_id, arm_joint_ids):
    """Run IK and return final EE position."""
    jacp = np.zeros((3, model.nv))
    for _ in range(IK_STEPS):
        mujoco.mj_forward(model, data)
        error = target_pos - data.site_xpos[ee_site_id]
        if np.linalg.norm(error) < 5e-4:
            break
        err_norm = np.linalg.norm(error)
        if err_norm > MAX_DPOS:
            error = error * MAX_DPOS / err_norm
        mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)
        J = jacp[:, arm_joint_ids]
        JJT = J @ J.T + DAMPING * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)
        for i, jid in enumerate(arm_joint_ids):
            data.qpos[jid] += dq[i]
            lo, hi = model.jnt_range[jid]
            data.qpos[jid] = np.clip(data.qpos[jid], lo, hi)
    for jid in arm_joint_ids:
        data.ctrl[jid] = data.qpos[jid]
    mujoco.mj_forward(model, data)
    return data.site_xpos[ee_site_id].copy()


# ---- Tests ----

def test_model_loads():
    """MJCF loads without errors and has expected structure."""
    model, data = load_model()
    assert model.nbody == 12, f"Expected 12 bodies, got {model.nbody}"
    assert model.njnt == 8, f"Expected 8 joints, got {model.njnt}"
    assert model.nmesh == 10, f"Expected 10 meshes, got {model.nmesh}"
    assert model.nu == 8, f"Expected 8 actuators, got {model.nu}"
    assert model.nmocap == 1, f"Expected 1 mocap body, got {model.nmocap}"
    print("  PASS  model_loads")


def test_joint_names():
    """All expected joints exist with correct types."""
    model, _ = load_model()
    expected_hinge = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    expected_slide = ["joint7", "joint8"]
    for name in expected_hinge:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"Joint '{name}' not found"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, f"{name} should be hinge"
    for name in expected_slide:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"Joint '{name}' not found"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE, f"{name} should be slide"
    print("  PASS  joint_names")


def test_keyframes_exist():
    """Home and zero keyframes exist."""
    model, _ = load_model()
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    zero_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "zero")
    assert home_id >= 0, "Home keyframe not found"
    assert zero_id >= 0, "Zero keyframe not found"
    print("  PASS  keyframes_exist")


def test_ee_site_exists():
    """End-effector site exists."""
    model, _ = load_model()
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    assert ee_id >= 0, "ee_site not found"
    print("  PASS  ee_site_exists")


def test_mocap_target_exists():
    """IK target mocap body exists."""
    model, _ = load_model()
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
    assert tid >= 0, "ik_target body not found"
    print("  PASS  mocap_target_exists")


def test_home_pose_ee_position():
    """At home pose, EE is above the base and at a reasonable height."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee = data.site_xpos[ee_id]
    # EE should be forward (x > 0.3) and elevated (z > 0.3)
    assert ee[2] > 0.3, f"EE too low at home: z={ee[2]:.3f}"
    assert ee[0] > 0.3, f"EE too far back at home: x={ee[0]:.3f}"
    # Should be near center in Y
    assert abs(ee[1]) < 0.05, f"EE off-center at home: y={ee[1]:.3f}"
    print(f"  PASS  home_pose_ee_position  (EE at {ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")


def test_arm_stays_connected():
    """All bodies remain connected (no flying apart) after simulation steps."""
    model, data = load_model()
    reset_home(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    # Check all body positions are within reasonable bounds
    for i in range(model.nbody):
        pos = data.xpos[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name in ("world", "ik_target"):
            continue
        dist = np.linalg.norm(pos)
        assert dist < 2.0, f"Body '{name}' flew away: pos={pos}, dist={dist:.3f}"
        assert pos[2] > -0.1, f"Body '{name}' fell through floor: z={pos[2]:.3f}"
    print("  PASS  arm_stays_connected")


def test_gravity_compensation():
    """Arm holds home pose under gravity (doesn't collapse)."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_start = data.site_xpos[ee_id].copy()

    # Run 1000 steps (~2 seconds)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    ee_end = data.site_xpos[ee_id].copy()
    drift = np.linalg.norm(ee_end - ee_start)
    assert drift < 0.1, f"EE drifted {drift:.3f}m under gravity (should hold pose)"
    print(f"  PASS  gravity_compensation  (drift: {drift:.4f}m)")


def test_ik_reaches_forward():
    """IK solver can reach a target in front of the arm."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))
    target = np.array([0.4, 0.0, 0.3])

    ee_final = solve_ik(model, data, target, ee_id, arm_ids)
    err = np.linalg.norm(ee_final - target)
    assert err < 0.02, f"IK failed to reach forward target: err={err:.4f}m"
    print(f"  PASS  ik_reaches_forward  (err: {err:.4f}m)")


def test_ik_reaches_left():
    """IK solver can reach a target to the left."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))
    target = np.array([0.2, 0.3, 0.3])

    ee_final = solve_ik(model, data, target, ee_id, arm_ids)
    err = np.linalg.norm(ee_final - target)
    assert err < 0.02, f"IK failed to reach left target: err={err:.4f}m"
    print(f"  PASS  ik_reaches_left  (err: {err:.4f}m)")


def test_ik_reaches_right():
    """IK solver can reach a target to the right."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))
    target = np.array([0.2, -0.3, 0.3])

    ee_final = solve_ik(model, data, target, ee_id, arm_ids)
    err = np.linalg.norm(ee_final - target)
    assert err < 0.02, f"IK failed to reach right target: err={err:.4f}m"
    print(f"  PASS  ik_reaches_right  (err: {err:.4f}m)")


def test_ik_reaches_high():
    """IK solver can reach a target above the arm."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))
    target = np.array([0.15, 0.0, 0.55])

    ee_final = solve_ik(model, data, target, ee_id, arm_ids)
    err = np.linalg.norm(ee_final - target)
    assert err < 0.02, f"IK failed to reach high target: err={err:.4f}m"
    print(f"  PASS  ik_reaches_high  (err: {err:.4f}m)")


def test_ik_respects_joint_limits():
    """IK solution stays within joint limits."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))
    target = np.array([0.4, 0.2, 0.2])

    solve_ik(model, data, target, ee_id, arm_ids)

    for jid in range(model.njnt):
        lo, hi = model.jnt_range[jid]
        val = data.qpos[jid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        assert lo - 0.001 <= val <= hi + 0.001, \
            f"Joint '{name}' out of limits: {val:.4f} not in [{lo:.3f}, {hi:.3f}]"
    print("  PASS  ik_respects_joint_limits")


def test_ik_unreachable_target_doesnt_explode():
    """IK with an unreachable target doesn't produce NaN or crash."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))
    # Way out of reach
    target = np.array([2.0, 2.0, 2.0])

    ee_final = solve_ik(model, data, target, ee_id, arm_ids)
    assert not np.any(np.isnan(ee_final)), "IK produced NaN"
    assert not np.any(np.isnan(data.qpos)), "qpos has NaN"
    # Should have moved toward the target direction at least
    print(f"  PASS  ik_unreachable_doesnt_explode  (EE at {ee_final[0]:.3f}, {ee_final[1]:.3f}, {ee_final[2]:.3f})")


def test_homing_converges():
    """Simulating with ctrl=HOME from a non-home pose converges to home."""
    model, data = load_model()
    # Start at zero pose
    zero_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "zero")
    mujoco.mj_resetDataKeyframe(model, data, zero_id)

    # Set ctrl to home
    for i in range(8):
        data.ctrl[i] = HOME[i]

    # Simulate for 2 seconds
    for _ in range(1000):
        mujoco.mj_step(model, data)

    # Check joints are near home
    max_err = 0
    for i in range(6):
        err = abs(data.qpos[i] - HOME[i])
        max_err = max(max_err, err)
    assert max_err < 0.15, f"Homing didn't converge: max joint error={max_err:.3f} rad"
    print(f"  PASS  homing_converges  (max joint err: {max_err:.4f} rad)")


def test_simulation_stable_over_time():
    """Run 5000 steps with IK active — no NaN, no explosion."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))

    targets = [
        np.array([0.4, 0.0, 0.3]),
        np.array([0.2, 0.3, 0.4]),
        np.array([0.3, -0.2, 0.2]),
    ]

    for t_idx, target in enumerate(targets):
        solve_ik(model, data, target, ee_id, arm_ids)
        for _ in range(500):
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), f"NaN in qpos after target {t_idx}"
        assert not np.any(np.isnan(data.xpos)), f"NaN in xpos after target {t_idx}"

    print("  PASS  simulation_stable_over_time")


def test_rapid_retarget_no_nan():
    """Rapidly switching IK targets (simulating user clicking fast) stays stable."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))

    targets = [
        np.array([0.4, 0.1, 0.3]),
        np.array([0.2, -0.3, 0.25]),
        np.array([0.5, 0.0, 0.15]),
        np.array([0.15, 0.2, 0.5]),
        np.array([0.3, -0.1, 0.35]),
    ]

    for target in targets:
        # Partial IK (only 20 steps — simulating interruption mid-solve)
        jacp = np.zeros((3, model.nv))
        for _ in range(20):
            mujoco.mj_forward(model, data)
            error = target - data.site_xpos[ee_id]
            err_norm = np.linalg.norm(error)
            if err_norm > 0.05:
                error = error * 0.05 / err_norm
            mujoco.mj_jacSite(model, data, jacp, None, ee_id)
            J = jacp[:, arm_ids]
            JJT = J @ J.T + 1e-4 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            for i, jid in enumerate(arm_ids):
                data.qpos[jid] += dq[i]
                lo, hi = model.jnt_range[jid]
                data.qpos[jid] = np.clip(data.qpos[jid], lo, hi)

        # A few physics steps between retargets
        for jid in arm_ids:
            data.ctrl[jid] = data.qpos[jid]
        for _ in range(50):
            mujoco.mj_step(model, data)

        assert not np.any(np.isnan(data.qpos)), "NaN in qpos during rapid retarget"
        assert not np.any(np.isnan(data.xpos)), "NaN in xpos during rapid retarget"

    print("  PASS  rapid_retarget_no_nan")


def test_direct_retarget_sequence():
    """IK directly from one target to another without homing."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))

    targets = [
        np.array([0.4, 0.1, 0.3]),
        np.array([0.2, -0.2, 0.25]),
        np.array([0.5, 0.0, 0.35]),
        np.array([0.15, 0.3, 0.4]),
    ]

    for i, target in enumerate(targets):
        ee_final = solve_ik(model, data, target, ee_id, arm_ids)
        for _ in range(200):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        ee_after = data.site_xpos[ee_id].copy()
        err = np.linalg.norm(ee_after - target)
        assert err < 0.05, f"Failed target {i}: err={err:.4f}m"
        assert not np.any(np.isnan(data.qpos)), f"NaN at target {i}"

    print(f"  PASS  direct_retarget_sequence  (4 targets reached)")


def test_ik_from_zero_pose():
    """IK works starting from zero pose (not just home)."""
    model, data = load_model()
    zero_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "zero")
    mujoco.mj_resetDataKeyframe(model, data, zero_id)
    mujoco.mj_forward(model, data)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))
    target = np.array([0.3, 0.0, 0.3])

    ee_final = solve_ik(model, data, target, ee_id, arm_ids)
    err = np.linalg.norm(ee_final - target)
    assert err < 0.05, f"IK from zero pose failed: err={err:.4f}m"
    assert not np.any(np.isnan(data.qpos)), "NaN after IK from zero"
    print(f"  PASS  ik_from_zero_pose  (err: {err:.4f}m)")


def test_back_and_forth_stability():
    """Moving between two targets repeatedly stays stable."""
    model, data = load_model()
    reset_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_ids = list(range(6))

    t1 = np.array([0.4, 0.2, 0.3])
    t2 = np.array([0.2, -0.2, 0.35])

    for cycle in range(6):
        target = t1 if cycle % 2 == 0 else t2
        solve_ik(model, data, target, ee_id, arm_ids)
        for _ in range(200):
            mujoco.mj_step(model, data)

        assert not np.any(np.isnan(data.qpos)), f"NaN at cycle {cycle}"
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name in ("world", "ik_target"):
                continue
            assert np.linalg.norm(data.xpos[i]) < 2.0, f"Body '{name}' flew away at cycle {cycle}"

    print("  PASS  back_and_forth_stability")


# ---- Runner ----

if __name__ == "__main__":
    tests = [
        test_model_loads,
        test_joint_names,
        test_keyframes_exist,
        test_ee_site_exists,
        test_mocap_target_exists,
        test_home_pose_ee_position,
        test_arm_stays_connected,
        test_gravity_compensation,
        test_ik_reaches_forward,
        test_ik_reaches_left,
        test_ik_reaches_right,
        test_ik_reaches_high,
        test_ik_respects_joint_limits,
        test_ik_unreachable_target_doesnt_explode,
        test_homing_converges,
        test_simulation_stable_over_time,
        test_rapid_retarget_no_nan,
        test_direct_retarget_sequence,
        test_ik_from_zero_pose,
        test_back_and_forth_stability,
    ]

    print(f"\nRunning {len(tests)} smoke tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 50}")
    exit(0 if failed == 0 else 1)
