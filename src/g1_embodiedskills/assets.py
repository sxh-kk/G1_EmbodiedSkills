"""Generate a local robot variant; upstream meshes are never redistributed."""
from pathlib import Path
import xml.etree.ElementTree as ET
from .io import sha256, write_json

GRIPPER_JOINTS = ["right_hand_index_0_joint", "right_hand_middle_0_joint"]
TCP_OFFSET = [.085, 0., 0.]


def box_link(name, size, mass, center=(0., 0., 0.), color="0.2 0.25 0.3 1"):
    link = ET.Element("link", name=name)
    inertial = ET.SubElement(link, "inertial")
    xyz = " ".join(map(str, center))
    ET.SubElement(inertial, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    a, b, c = size
    ET.SubElement(inertial, "inertia", ixx=str(mass*(b*b+c*c)/12),
        iyy=str(mass*(a*a+c*c)/12), izz=str(mass*(a*a+b*b)/12), ixy="0", ixz="0", iyz="0")
    for kind in ["visual", "collision"]:
        element = ET.SubElement(link, kind)
        ET.SubElement(element, "origin", xyz=xyz, rpy="0 0 0")
        ET.SubElement(ET.SubElement(element, "geometry"), "box", size=" ".join(map(str, size)))
        if kind == "visual":
            ET.SubElement(ET.SubElement(element, "material", name=name+"_material"), "color", rgba=color)
    return link


def build_robot(sonic, output):
    sonic, output = Path(sonic).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = sonic / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
    tree = ET.parse(original)
    root = tree.getroot()
    # Remove the fixed right-hand subtree, retaining the 29 body joints.
    removed = {"right_hand_palm_link"}
    while True:
        children = {j.find("child").get("link") for j in root.findall("joint")
                    if j.find("parent").get("link") in removed}
        new = children - removed
        if not new:
            break
        removed |= new
    for node in list(root):
        if (node.tag == "link" and node.get("name") in removed) or (
            node.tag == "joint" and node.find("child").get("link") in removed):
            root.remove(node)
    palm = box_link("right_gripper_palm", [.04, .06, .035], .16, [.02, 0., 0.])
    root.append(palm)
    mount = ET.SubElement(root, "joint", name="right_gripper_mount", type="fixed")
    ET.SubElement(mount, "parent", link="right_wrist_yaw_link")
    ET.SubElement(mount, "child", link="right_gripper_palm")
    ET.SubElement(mount, "origin", xyz="0 0 0", rpy="0 0 0")
    for index, sign in enumerate([1, -1]):
        name = f"right_gripper_finger_{index}"
        root.append(box_link(name, [.08, .012, .024], .035, [.04, 0., 0.], "0.15 0.15 0.15 1"))
        joint = ET.SubElement(root, "joint", name=GRIPPER_JOINTS[index], type="prismatic")
        ET.SubElement(joint, "parent", link="right_gripper_palm")
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz=f"0.04 {sign*.008} 0", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=f"0 {sign} 0")
        ET.SubElement(joint, "limit", lower="0", upper="0.04", effort="20", velocity="0.15")
        ET.SubElement(joint, "dynamics", damping="2", friction="0")
    # Resolve package URLs locally, preserving source provenance outside Git.
    package = sonic / "gear_sonic/data/assets/robot_description"
    for mesh in root.iter("mesh"):
        name = mesh.get("filename")
        if name.startswith("package://robot_description/"):
            mesh.set("filename", str(package / name.removeprefix("package://robot_description/")))
    path = output / "g1_parallel_gripper.urdf"
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    movable = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    assert len(movable) == 31 and set(GRIPPER_JOINTS) <= set(movable)
    write_json(output / "asset_manifest.json", dict(source_sha256=sha256(original), generated_sha256=sha256(path),
        body_dof=29, gripper_dof=2, gripper_mass_kg=.23, tcp_offset=TCP_OFFSET,
        opening_m=[.004, .084], added_joint_names=GRIPPER_JOINTS, removed_links=sorted(removed)))
    return path
