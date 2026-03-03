import pickle
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import numpy as np
import os

# bone_vectors: [body/lhand/rhand][njoints] ('pHipOrigin', 'jRightHip'): 0.09663602709770203
# body_global_transform.pkl (nframes, 4, 4)
# body_joint_orientations.pkl (nframes, 23, 6)
# joint_positions.pkl (nframes, 73, 3)
# joint_states.pkl [nobjects](nframes, 1) # 旋转部分用弧度表示(eg. laptop)，棱柱部分用米表示(draw)
# hand_joint_orientations.pkl (nframes, 40, 6)
# head_tips.pkl (nframes, 3)

# Load bone vectors
scene_numbers = [11,29,79,89,98,99,108,110,113,114,117,149,184]
for scene_number in scene_numbers:
    with open(f'/home/kimyw/github/sim/InterMimic/intermimic/data/assets/parahome/seq/s{scene_number}/bone_vectors.pkl', 'rb') as f:
        bone_vectors = pickle.load(f)

    # Joint information
    body_vector = bone_vectors.get('body', {})
    lhand_vector = bone_vectors.get('lhand', {})
    rhand_vector = bone_vectors.get('rhand', {})

    # Body part classification for size and fromto rules
    BODY_PART_CONFIG = {
        # Spine
        'jL5S1': {'type': 'capsule', 'size': 0.03, 'fromto_start': (0, 0, 0.02), 'density': 1000},
        'jL4L3': {'type': 'capsule', 'size': 0.03, 'fromto_start': (0, 0, 0), 'density': 1000},
        'jL1T12': {'type': 'capsule', 'size': 0.03, 'fromto_start': (0, 0, 0), 'density': 1000},
        'jT9T8': {'type': 'sphere', 'size': 0.05011, 'pos_offset': (0, 0, 0.05), 'density': 1000},
        'jT1C7': {'type': 'capsule', 'size': 0.02, 'fromto_start': (0, 0, 0.03), 'density': 1000},
        'jC1Head': {'type': 'cylinder', 'size': 0.05, 'fromto_end': (0, 0, 0.15), 'density': 1000},

        # Arms - Shoulders
        'jRightT4Shoulder': {'type': 'capsule', 'size': 0.04, 'fromto_start': (0, -0.05, 0), 'density': 1000},
        'jLeftT4Shoulder': {'type': 'capsule', 'size': 0.04, 'fromto_start': (0, 0.05, 0), 'density': 1000},

        # Arms - Upper
        'jRightShoulder': {'type': 'capsule', 'size': 0.036, 'fromto_start': (0, -0.05, 0), 'density': 1000},
        'jLeftShoulder': {'type': 'capsule', 'size': 0.036, 'fromto_start': (0, 0.05, 0), 'density': 1000},

        # Arms - Forearms
        'jRightElbow': {'type': 'capsule', 'size': 0.034, 'fromto_start': (0, -0.05, 0), 'density': 1000},
        'jLeftElbow': {'type': 'capsule', 'size': 0.034, 'fromto_start': (0, 0.05, 0), 'density': 1000},

        # Wrists - Box shape
        'jRightWrist': {'type': 'box', 'size': (0.035, 0.03, 0.01), 'density': 1000},
        'jLeftWrist': {'type': 'box', 'size': (0.035, 0.03, 0.01), 'density': 1000},

        # Legs
        'jRightHip': {'type': 'capsule', 'size': 0.0605, 'fromto_start': (0, 0, -0.08), 'density': 2040.816327},
        'jLeftHip': {'type': 'capsule', 'size': 0.0605, 'fromto_start': (0, 0, -0.08), 'density': 2040.816327},
        'jRightKnee': {'type': 'capsule', 'size': 0.0533, 'fromto_start': (0, 0, -0.1), 'density': 1234.567901},
        'jLeftKnee': {'type': 'capsule', 'size': 0.0533, 'fromto_start': (0, 0, -0.1), 'density': 1234.567901},

        # Feet - Box shape
        'jRightAnkle': {'type': 'box', 'size': (0.1, 0.05, 0.03), 'pos_offset': (0.04, 0, -0.03), 'density': 1000},
        'jLeftAnkle': {'type': 'box', 'size': (0.1, 0.05, 0.03), 'pos_offset': (0.04, 0, -0.03), 'density': 1000},
        'jRightBallFoot': {'type': 'box', 'size': (0.01, 0.05, 0.01), 'pos_offset': (0, 0, 0.02), 'density': 1000},
        'jLeftBallFoot': {'type': 'box', 'size': (0.01, 0.05, 0.01), 'pos_offset': (0, 0, 0.02), 'density': 1000},
    }

    # Default config for finger joints and unspecified parts
    DEFAULT_FINGER_CONFIG = {'type': 'capsule', 'size': 0.005, 'fromto_start': (0, 0, 0), 'density': 1000}

    def get_joint_config(joint_name):
        """Get configuration for a joint, with finger default fallback."""
        if joint_name in BODY_PART_CONFIG:
            return BODY_PART_CONFIG[joint_name]
        # Default for fingers and unspecified joints
        return DEFAULT_FINGER_CONFIG

    def create_body(parent, name, bone_vector, children_dict, child_is_box_joint=False, geom_type='capsule'):
        """
        Create a body element with joints and geometry.

        Args:
            parent: Parent XML element
            name: Name of the body/joint
            bone_vector: 3D numpy array representing position offset from parent
            children_dict: Dictionary of children for this body (to get fromto end points)
            child_is_box_joint: True if the child joint has a box geometry (wrist/ankle)
            geom_type: Type of geometry (overridden by config)
        """
        config = get_joint_config(name)

        # Position is the bone vector itself (offset from parent)
        pos_str = ' '.join(map(str, bone_vector))

        # Create body element
        body = SubElement(parent, 'body', name=name, pos=pos_str)

        # Get configuration
        geom_type = config.get('type', 'capsule')
        size = config['size']
        density = config.get('density', 1000)

        # Create geometry based on type
        if geom_type == 'capsule':
            # Capsule with fromto
            fromto_start = config.get('fromto_start', (0, 0, 0))

            # Calculate fromto end: use child position if available
            if children_dict:
                # Use the first child's bone vector as the endpoint
                first_child_name = list(children_dict.keys())[0]
                first_child_vector = list(children_dict.values())[0]
                fromto_end = list(first_child_vector)

                # If child is a box joint (wrist/ankle), adjust endpoint to avoid overlap
                if child_is_box_joint:
                    child_config = get_joint_config(first_child_name)
                    if child_config.get('type') == 'box':
                        # Use hardcoded gaps from reference s22 file
                        # For wrists (Y-axis), ankles (Z-axis)
                        if abs(fromto_end[1]) > 0.01:  # Y-axis dominant (arms)
                            # Wrist gap: 0.066 (hardcoded from reference)
                            adjustment = 0.066
                            fromto_end[1] = fromto_end[1] + (adjustment if fromto_end[1] < 0 else -adjustment)
                        elif abs(fromto_end[2]) > 0.01:  # Z-axis dominant (legs)
                            # Ankle gap: 0.064 (hardcoded from reference)
                            adjustment = 0.064
                            fromto_end[2] = fromto_end[2] + (adjustment if fromto_end[2] < 0 else -adjustment)

                fromto_end = tuple(fromto_end)
            else:
                # No children, just extend along bone direction
                fromto_end = tuple(bone_vector * 0.5)

            fromto_str = f"{fromto_start[0]} {fromto_start[1]} {fromto_start[2]} {fromto_end[0]} {fromto_end[1]} {fromto_end[2]}"

            SubElement(body, 'geom',
                    type='capsule',
                    contype='1',
                    conaffinity='1',
                    density=str(density),
                    fromto=fromto_str,
                    size=str(size))

        elif geom_type == 'sphere':
            pos_offset = config.get('pos_offset', (0, 0, 0))
            pos_offset_str = f"{pos_offset[0]:.4f} {pos_offset[1]:.4f} {pos_offset[2]:.6f}"
            SubElement(body, 'geom',
                    type='sphere',
                    contype='1',
                    conaffinity='1',
                    density=str(density),
                    size=str(size),
                    pos=pos_offset_str)

        elif geom_type == 'cylinder':
            fromto_end = config.get('fromto_end', (0, 0, 0.15))
            fromto_str = f"0 0 0 {fromto_end[0]} {fromto_end[1]} {fromto_end[2]}"
            SubElement(body, 'geom',
                    type='cylinder',
                    contype='1',
                    conaffinity='1',
                    density=str(density),
                    fromto=fromto_str,
                    size=str(size))

        elif geom_type == 'box':
            pos_offset = config.get('pos_offset', (0, 0, 0))
            pos_str = f"{pos_offset[0]} {pos_offset[1]} {pos_offset[2]}"
            size_tuple = size if isinstance(size, tuple) else (size, size, size)
            size_str = f"{size_tuple[0]} {size_tuple[1]} {size_tuple[2]}"
            SubElement(body, 'geom',
                    density=str(density),
                    type='box',
                    pos=pos_str,
                    size=size_str,
                    quat='1.0 0 0 0')

        # Add 3 hinge joints for rotation around each axis
        SubElement(body, 'joint', name=f"{name}_x", type='hinge', pos='0 0 0', axis='1 0 0',
                stiffness='500', damping='500', armature='0.02', range='-180.0000 180.0000')
        SubElement(body, 'joint', name=f"{name}_y", type='hinge', pos='0 0 0', axis='0 1 0',
                stiffness='500', damping='500', armature='0.02', range='-180.0000 180.0000')
        SubElement(body, 'joint', name=f"{name}_z", type='hinge', pos='0 0 0', axis='0 0 1',
                stiffness='500', damping='500', armature='0.02', range='-180.0000 180.0000')

        return body

    # Create MuJoCo XML structure
    mujoco = Element('mujoco', model='humanoid')

    # Compiler and options
    SubElement(mujoco, 'compiler', coordinate='local')
    SubElement(mujoco, 'statistic', extent='2', center='0 0 1')
    SubElement(mujoco, 'option', timestep='0.00555')

    # Default settings
    default = SubElement(mujoco, 'default')
    SubElement(default, 'motor', ctrlrange='-1 1', ctrllimited='true')
    SubElement(default, 'geom', type='capsule', condim='1', friction='1.0 0.05 0.05',
            solimp='.9 .99 .003', solref='.015 1')
    SubElement(default, 'joint', type='hinge', damping='0.1', stiffness='5',
            armature='.007', limited='true', solimplimit='0 .99 .01')
    SubElement(default, 'site', size='.04', group='3')

    force_torque = SubElement(default, 'default', **{'class': 'force-torque'})
    SubElement(force_torque, 'site', type='box', size='.01 .01 .02', rgba='1 0 0 1')

    touch = SubElement(default, 'default', **{'class': 'touch'})
    SubElement(touch, 'site', type='capsule', rgba='0 0 1 .3')

    # Asset section
    asset = SubElement(mujoco, 'asset')
    SubElement(asset, 'texture', type='skybox', builtin='gradient',
            rgb1='.4 .5 .6', rgb2='0 0 0', width='100', height='100')
    SubElement(asset, 'texture', builtin='flat', height='1278', mark='cross',
            markrgb='1 1 1', name='texgeom', random='0.01',
            rgb1='0.8 0.6 0.4', rgb2='0.8 0.6 0.4', type='cube', width='127')
    SubElement(asset, 'texture', builtin='checker', height='100', name='texplane',
            rgb1='0 0 0', rgb2='0.8 0.8 0.8', type='2d', width='100')
    SubElement(asset, 'material', name='MatPlane', reflectance='0.5', shininess='1',
            specular='1', texrepeat='60 60', texture='texplane')
    SubElement(asset, 'material', name='geom', texture='texgeom', texuniform='true')

    # Worldbody
    worldbody = SubElement(mujoco, 'worldbody')
    SubElement(worldbody, 'light', cutoff='100', diffuse='1 1 1', dir='-0 0 -1.3',
            directional='true', exponent='1', pos='0 0 1.3', specular='.1 .1 .1')
    SubElement(worldbody, 'geom', conaffinity='1', condim='3', name='floor',
            pos='0 0 0', rgba='0.8 0.9 0.8 1', size='100 100 .2',
            type='plane', material='MatPlane')

    # Build body hierarchy
    bodies = {}
    joint_names = []  # Track all joints for actuator generation

    # Process all bone vectors (body, left hand, right hand)
    all_vectors = {'body': body_vector, 'lhand': lhand_vector, 'rhand': rhand_vector}

    for section_name, vector_dict in all_vectors.items():
        for parent_name, children_dict in vector_dict.items():
            # Create parent body if it doesn't exist
            if parent_name not in bodies:
                if parent_name == 'pHipOrigin':
                    # Root body (Pelvis) - placed at origin with a freejoint
                    bodies[parent_name] = SubElement(worldbody, 'body', name='pHipOrigin', pos='0 0 0')
                    SubElement(bodies[parent_name], 'freejoint', name='pHipOrigin')
                    # Pelvis sphere geometry
                    SubElement(bodies[parent_name], 'geom', type='sphere', contype='1', conaffinity='1',
                            density='4629.6296296296305', size='0.07011', pos='0.0000 0.0000 -0.0000')
                else:
                    # Parent should have been created as a child before, if not, skip
                    print(f"Warning: Parent '{parent_name}' not found in bodies dict. Skipping.")
                    continue

            # Create all children of this parent
            for child_name, bone_vector in children_dict.items():
                # Skip fingertip bodies (they start with 'p' and are just position markers)
                if child_name.startswith('p'):
                    continue

                # Get children of this child for fromto calculation
                child_children = {}
                for section in all_vectors.values():
                    if child_name in section:
                        child_children = section[child_name]
                        break

                # Check if any child is a box joint (wrist/ankle)
                child_is_box = False
                if child_children:
                    first_child_name = list(child_children.keys())[0]
                    child_config = get_joint_config(first_child_name)
                    if child_config.get('type') == 'box':
                        child_is_box = True

                bodies[child_name] = create_body(bodies[parent_name], child_name,
                                                bone_vector, child_children, child_is_box)

                # Track joints for actuators
                joint_names.extend([f"{child_name}_x", f"{child_name}_y", f"{child_name}_z"])

    # Actuator section
    actuator = SubElement(mujoco, 'actuator')
    for joint_name in joint_names:
        SubElement(actuator, 'motor', name=joint_name, joint=joint_name, gear='500')

    # Format XML output with comment
    xml_str = '<!-- change the damping from 50 to 500 -->\n'
    xml_str += xml.dom.minidom.parseString(tostring(mujoco)).toprettyxml(indent="  ")

    # Remove extra XML declaration from minidom
    lines = xml_str.split('\n')
    xml_str = '\n'.join([lines[0]] + [line for line in lines[1:] if not line.strip().startswith('<?xml')])

    # Ensure output directory exists
    output_dir = f"skillmimic/data/assets/mjcf/parahome"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/s{scene_number}.xml"
    with open(output_path, "w") as f:
        f.write(xml_str)

    print(f"XML file '{output_path}' created successfully.")
