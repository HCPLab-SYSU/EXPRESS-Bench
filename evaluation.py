import numpy as np


def score(results):
    C, C_star, p_path, l_path, d_T = [], [], [], [], []
    for result in results:
        if result["path_len"] != float("inf"):
            EAC = result["EAC"].replace("Your mark:", "").strip()
            grd, acc = float(EAC.split(",")[0]), int(EAC.split(",")[1])
            C.append(grd*acc)
            C_star.append(acc)
            p_path.append(result["path_len"])
            l_path.append(result["geodesic_distance"])
        if result["goal_dis"] != float("inf"):
            d_T.append(result["goal_dis"])

    weight_path = l_path / np.maximum(p_path, l_path)
    C_avg = np.mean(100.0 * (np.clip(C, 0, 5) / 5))
    C_star_avg = np.mean(100.0 * (np.clip(C_star, 0, 5) / 5))
    E_path = np.mean(100.0 * (np.clip(C, 0, 5) / 5) * weight_path)
    d_T_avg = np.mean(d_T)

    return C_avg, C_star_avg, E_path, d_T_avg

