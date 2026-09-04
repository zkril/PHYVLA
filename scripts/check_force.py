import numpy as np

data = np.load(
    "outputs/physics/task02_demo00_step6_fixed.npz"
)

contacts = data["contacts"]
mask = data["contact_mask"] > 0.5

normal_force = contacts[..., 13][mask]
tangential_force = contacts[..., 14][mask]
force_norm = contacts[..., 15][mask]
torque_norm = contacts[..., 16][mask]

print("有效接触数:", int(mask.sum()))
print("最大法向力:", float(normal_force.max()))
print("平均法向力:", float(normal_force.mean()))
print("最大切向力:", float(tangential_force.max()))
print("最大总力:", float(force_norm.max()))
print("最大力矩:", float(torque_norm.max()))

print(
    "非零力比例:",
    float(np.mean(force_norm > 1e-8)),
)
