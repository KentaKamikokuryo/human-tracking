from typing import List
from pyparsing import Optional
from box import Box, Body, Head
from mmdet.apis import init_detector, inference_detector


model_names = ["YOLOX", "DINO", "Pedestron"]

class ModelBoxDetectionMMDet:

    def __init__(
        self,
        path_model, 
        threshold: float = .6, 
        model_name="DINO"
        ):

        self.model_name = model_name
        self.threshold = threshold
        
        config_path = f"{path_model}\\mmdetection\\configs\\"
        checkpoint_path = f"{path_model}\\mmdetection\\checkpoint\\"

        model_dict = {
            "YOLOX": {"config": config_path + "yolox\\yolox_x_8xb8-300e_coco.py",
                        "checkpoint": checkpoint_path + "yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"},
            "DINO": {"config": config_path + "dino\\dino-5scale_swin-l_8xb2-36e_coco.py",
                        "checkpoint": checkpoint_path + "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"},
            "Pedestron": {"config": config_path + "faster_rcnn\\faster-rcnn_x101-64x4d_fpn_ms-3x_coco.py",
                            "checkpoint": checkpoint_path + "faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth"},
        }

        self.config_file = model_dict[self.model_name]["config"]
        self.checkpoint_file = model_dict[self.model_name]["checkpoint"]

        self.model = init_detector(config=self.config_file, checkpoint=self.checkpoint_file)
        self.model.CLASSES = ["person"]  # we interested in only person

    def __call__(self, image) -> List[Box]:

        result = inference_detector(model=self.model, imgs=image)
        result_boxes = self._process(infer_result=result)

        return result_boxes

    def _process(self, infer_result):

        xyxys = infer_result.pred_instances.bboxes.tolist()
        confs = infer_result.pred_instances.scores.tolist()
        clses = infer_result.pred_instances.labels.tolist()
        n_boxes = len(xyxys)

        result_boxes: List[Box] = []

        if n_boxes > 0:
            
            for n in range(n_boxes):
                
                cls = clses[n]
                conf = confs[n]
                
                if cls == 0 and conf >= self.threshold:
                    
                    xyxy = xyxys[n]
                    x_min = xyxy[0]
                    y_min = xyxy[1]
                    x_max = xyxy[2]
                    y_max = xyxy[3]
                    
                    result_boxes.append(
                        Box(
                            trackid=0,
                            classid=0, # class is body
                            score=float(conf),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=x_min // x_max,
                            cy=y_min // y_max,
                            is_used=False
                        )
                    )

        return result_boxes