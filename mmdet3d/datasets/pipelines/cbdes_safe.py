from typing import Any, Dict, Tuple

import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RadarSanitize:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "radar" in data:
            radar_obj = data["radar"]
            has_ops = all(
                hasattr(radar_obj, attr)
                for attr in ("rotate", "translate", "scale")
            )
            has_flip = hasattr(radar_obj, "flip")
            has_range = hasattr(radar_obj, "in_range_bev")
            if not (has_ops and has_flip and has_range):
                # 移除占位或不完整的 radar 字段，防止下游算子误用
                data.pop("radar")
        return data


@PIPELINES.register_module()
class LoadBEVSegmentationSafe:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        # 延迟导入 NuScenesMap，避免在无 nuscenes 安装时出错（仅在有 location 时需要）
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.map_expansion.map_api import locations as LOCATIONS

        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        # 关键矩阵
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data.get("location", None)
        num_classes = len(self.classes)
        if location is None or location not in self.maps:
            data["gt_masks_bev"] = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
            return data

        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                idx = layer_names.index(layer_name)
                labels[k, masks[idx]] = 1

        data["gt_masks_bev"] = labels
        return data


