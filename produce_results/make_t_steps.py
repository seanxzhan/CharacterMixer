import numpy as np


def make_t_pose(num_frames):
    num_interp = num_frames - 1
    delta_t = 1 / num_interp
    lst_of_t = []
    lst_of_interp_steps = []
    for i in range(num_frames):
        lst_of_t.append(i * delta_t)
        lst_of_interp_steps.append(i)
    return lst_of_t, lst_of_interp_steps, num_interp


def make_inf_loop(num_frames):
    lst_of_t = []
    lst_of_interp_steps = []
    stay = 5
    num_interp = (num_frames - 2 * stay) / 2 + 1
    assert num_interp == np.floor(num_interp)
    num_interp = int(num_interp)
    delta_t = 1 / num_interp
    for _ in range(stay):
        lst_of_t.append(0)
        lst_of_interp_steps.append(0)
    for i in range(1, num_interp):
        lst_of_interp_steps.append(i)
        lst_of_t.append(i * delta_t)
    for _ in range(stay):
        lst_of_t.append(1)
        lst_of_interp_steps.append(num_interp)
    for i in reversed(range(1, num_interp)):
        lst_of_interp_steps.append(i)
        lst_of_t.append(i * delta_t)
    return lst_of_t, lst_of_interp_steps, num_interp


def make_2097_2091(num_frames):
    assert num_frames == 54
    lst_of_t = []
    lst_of_interp_steps = []
    num_interp = 20
    stay = 7
    interp_ts = []
    interp_steps = []
    delta_t = 1 / num_interp
    for i in range(0, num_interp + 1):
        interp_ts.append(i * delta_t)
        interp_steps.append(i)
    for i in range(3):
        lst_of_t.append(0)
        lst_of_interp_steps.append(0)
    for i in range(num_interp):
        lst_of_t.append(i * delta_t)
        lst_of_interp_steps.append(i)
    for i in range(stay):
        lst_of_t.append(1)
        lst_of_interp_steps.append(num_interp)
    for i in range(1, num_interp):
        lst_of_t.append(1 - i * delta_t)
        lst_of_interp_steps.append(num_interp - i)
    for i in range(num_frames - 3 - num_interp * 2 - stay + 1):
        lst_of_t.append(0)
        lst_of_interp_steps.append(0)
    return lst_of_t, lst_of_interp_steps, num_interp


def make_4010_1919(num_frames):
    assert num_frames == 35
    lst_of_t = []
    lst_of_interp_steps = []
    num_interp = num_frames - 1
    delta_t = 1 / num_interp
    for i in range(num_interp + 1):
        lst_of_t.append(i * delta_t)
        lst_of_interp_steps.append(i)
    lst_of_t = lst_of_t[::-1]
    lst_of_interp_steps = lst_of_interp_steps[::-1]
    return lst_of_t, lst_of_interp_steps, num_interp


def make_12852_12901(num_frames):
    assert num_frames == 78
    lst_of_t = []
    lst_of_interp_steps = []
    num_interp_1 = 30
    delta_t_1 = 1 / num_interp_1
    for i in range(num_interp_1 + 1):
        lst_of_interp_steps.append(i)
        lst_of_t.append(i * delta_t_1)
    for _ in range(49 - 31):
        lst_of_interp_steps.append(num_interp_1)
        lst_of_t.append(1)
    num_interp_2 = 78 - 49
    delta_t_2 = 1 / num_interp_2
    for i in reversed(range(num_interp_2)):
        lst_of_interp_steps.append(i)
        lst_of_t.append(i * delta_t_2)
    return lst_of_t, lst_of_interp_steps, max(num_interp_1, num_interp_2)


def make_16880_16827(num_frames):
    assert num_frames == 110
    lst_of_t = []
    lst_of_interp_steps = []
    num_interp = 40 + 1
    delta_t = 1 / num_interp
    for _ in range(30):
        lst_of_t.append(0)
        lst_of_interp_steps.append(0)
    for i in range(1, num_interp):
        lst_of_t.append(i * delta_t)
        lst_of_interp_steps.append(i)
    for _ in range(110 - 30 - num_interp + 1):
        lst_of_t.append(1)
        lst_of_interp_steps.append(num_interp)
    return lst_of_t, lst_of_interp_steps, num_interp


if __name__ == "__main__":
    lt, lis, ni = make_t_pose(11)
    print(lt)
    print(lis)
    print(ni)
    lt, lis, ni = make_inf_loop(44)
    print(lt)
    print(lis)
    print(len(lt))
    print(len(lis))
    print(ni)