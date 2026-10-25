import json
import os
import datasets

# 定义常量
_NUM_ANSWERS_PER_QUESTION = 10


class TextvqaLocal(datasets.GeneratorBasedBuilder):
    """TextVQA 本地数据加载脚本 (支持 train / test，包含 Rosetta OCR)"""

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("string"),
                "question_id": datasets.Value("int32"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
                "answers": datasets.Sequence(datasets.Value("string")),
                "ocr_tokens": datasets.Sequence(datasets.Value("string")),
                "ocr_boxes": datasets.Sequence(
                    datasets.Sequence(datasets.Value("float32"))
                ),
                "set_name": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        train_base = "/media/user/d/xyk/data1/textVQA/train"
        test_base = "/media/user/d/xyk/data1/textVQA/test"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotations_path": os.path.join(
                        train_base, "TextVQA_0.5.1_train.json"
                    ),
                    "ocr_path": os.path.join(
                        train_base, "TextVQA_Rosetta_OCR_v0.2_train.json"
                    ),
                    "images_dir": os.path.join(
                        train_base, "train_val_images/train_images"
                    ),
                    "default_set_name": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotations_path": os.path.join(
                        test_base, "TextVQA_0.5.1_test.json"
                    ),
                    "ocr_path": os.path.join(
                        test_base, "TextVQA_Rosetta_OCR_v0.2_test.json"
                    ),
                    "images_dir": os.path.join(
                        test_base, "test_images/test_images"
                    ),
                    "default_set_name": "test",
                },
            ),
        ]

    def _generate_examples(
        self,
        annotations_path,
        ocr_path,
        images_dir,
        default_set_name,
    ):
        # 1. 先读取 OCR 数据并建立索引，提高查询效率
        ocr_dict = {}
        if ocr_path is not None and os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_raw = json.load(f)["data"]
                ocr_dict = {item["image_id"]: item for item in ocr_raw}

        # 2. 读取标注文件
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]

        for idx, item in enumerate(data):
            image_id = item["image_id"]

            # 获取该图片对应的 OCR 信息
            current_ocr = ocr_dict.get(image_id, {})

            yield idx, {
                "image_id": image_id,
                "question_id": item["question_id"],
                "question": item["question"],
                "image": os.path.join(images_dir, f"{image_id}.jpg"),
                "answers": item.get("answers", [""] * _NUM_ANSWERS_PER_QUESTION),
                "ocr_tokens": current_ocr.get("ocr_tokens", []),
                "ocr_boxes": current_ocr.get("ocr_boxes", []),
                "set_name": item.get("set_name", default_set_name),
            }