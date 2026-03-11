# Package Renaming Map

The following files were renamed inside the evaluation package to improve clarity and professionalism without altering the original local source repository.

| Package Original Name | Package New Name |
|-----------------------|------------------|
| `02_deployment/hailo_live_pipeline_final.py` | `02_deployment/main_edge_inference_pipeline.py` |
| `02_deployment/hailo_live_pipeline.py` | `02_deployment/legacy_hailo_live_pipeline.py` |
| `02_deployment/hailo_live_pipeline_fastdepth.py` | `02_deployment/experimental_hailo_fastdepth_pipeline.py` |
| `02_deployment/hailo_live_pipeline_perp.py` | `02_deployment/experimental_hailo_perpendicular_pipeline.py` |
| `02_deployment/hailo_headless_pipeline.py` | `02_deployment/hailo_headless_inference_pipeline.py` |
| `02_deployment/test_hailo_models_final.py` | `02_deployment/validate_hailo_models.py` |
| `02_deployment/test_hailo_models.py` | `02_deployment/legacy_test_hailo_models.py` |
| `02_deployment/test_hailo_models_2.py` | `02_deployment/legacy_test_hailo_models_v2.py` |
| `02_deployment/test_hailo_models_3.py` | `02_deployment/legacy_test_hailo_models_v3.py` |
| `02_deployment/depth_test_hailo.py` | `02_deployment/validate_depth_estimation.py` |
| `02_deployment/depth_test_hailo_precision.py` | `02_deployment/validate_depth_precision.py` |
| `02_deployment/depth_test_hailo_scdepthv3.py` | `02_deployment/validate_depth_scdepthv3.py` |
| `02_deployment/depth_test_hailo_stereo.py` | `02_deployment/validate_depth_stereo.py` |
| `02_deployment/orchestrator.py` | `02_deployment/deployment_orchestrator.py` |
| `03_training/prepare_calib_yolo.py` | `03_training/export_calibration_yolo.py` |
| `03_training/prepare_calib_yolo_1.py` | `03_training/legacy_export_calibration_yolo_v1.py` |
| `03_training/prepare_calib_classifier.py` | `03_training/export_calibration_classifier.py` |
| `03_training/prepare_calib_classifier_1.py` | `03_training/legacy_export_calibration_classifier_v1.py` |
| `03_training/prepare_calib_lesion.py` | `03_training/export_calibration_lesion.py` |
| `03_training/prepare_calib_lesion_1.py` | `03_training/legacy_export_calibration_lesion_v1.py` |
| `03_training/train_classifier.py` | `03_training/train_disease_classifier.py` |
| `04_models/hef/lesion_segmentation_final.hef` | `04_models/hef/lesion_segmentation_unet.hef` |
| `04_models/hef/potato_classifier_final.hef` | `04_models/hef/disease_classifier_mobilenet.hef` |
| `04_models/hef/yolov8n_seg_final.hef` | `04_models/hef/leaf_segmentation_yolov8n.hef` |
| `04_models/onnx/potato_classifier.onnx` | `04_models/onnx/disease_classifier_mobilenet.onnx` |
| `04_models/onnx/potato_classifier_1.onnx` | `04_models/onnx/legacy_disease_classifier_v1.onnx` |
| `04_models/onnx/lesion_segmentation_1.onnx` | `04_models/onnx/legacy_lesion_segmentation_v1.onnx` |
| `04_models/onnx/yolov8n_seg.onnx` | `04_models/onnx/leaf_segmentation_yolov8n.onnx` |
| `05_configs/requirements.txt` | `05_configs/edge_requirements.txt` |
| `06_scripts/t.py` | `06_scripts/experimental_tensor_test.py` |
| `06_scripts/convert_all.sh` | `06_scripts/compile_all_to_hef.sh` |
| `06_scripts/convert_all_1.sh` | `06_scripts/legacy_compile_all_to_hef_v1.sh` |
