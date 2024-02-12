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

class YOLOX(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolox_x_body_head_hand_face_0076_0.5228_post_1x3x480x640_score015_iou080_box050.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """YOLOX

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """YOLOX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, x1, y1, x2, y2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]

        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self._input_shapes[0][self._w_index]),
                int(self._input_shapes[0][self._h_index]),
            )
        )
        resized_image = resized_image.transpose(self._swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )

        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    cx = x_min // x_max
                    cy = y_min // y_max
                    result_boxes.append(
                        Box(
                            trackid=0,
                            classid=int(box[1]),
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                            is_used=False,
                        )
                    )

        return result_boxes
