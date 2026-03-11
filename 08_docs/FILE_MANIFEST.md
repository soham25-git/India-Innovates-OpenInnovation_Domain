# Included File Manifest

This document maps the original repository files to their evaluated packaged paths.

| Original Path | Packaged Path |
|--------------|---------------|
| `camera_diagnostic.py` | `02_deployment/camera_diagnostic.py` |
| `validate_depth_estimation.py` | `02_deployment/validate_depth_estimation.py` |
| `depth_test_hailo_precision.py` | `02_deployment/validate_depth_precision.py` |
| `depth_test_hailo_scdepthv3.py` | `02_deployment/validate_depth_scdepthv3.py` |
| `depth_test_hailo_stereo.py` | `02_deployment/validate_depth_stereo.py` |
| `hailo_headless_inference_pipeline.py` | `02_deployment/hailo_headless_inference_pipeline.py` |
| `legacy_hailo_live_pipeline.py` | `02_deployment/legacy_hailo_live_pipeline.py` |
| `experimental_hailo_fastdepth_pipeline.py` | `02_deployment/experimental_hailo_fastdepth_pipeline.py` |
| `HAILO/final output/legacy_hailo_live_pipeline.py` | `02_deployment/main_edge_inference_pipeline.py` |
| `experimental_hailo_perpendicular_pipeline.py` | `02_deployment/experimental_hailo_perpendicular_pipeline.py` |
| `deployment_orchestrator.py` | `02_deployment/deployment_orchestrator.py` |
| `legacy_test_hailo_models.py` | `02_deployment/legacy_test_hailo_models.py` |
| `test_hailo_models_2.py` | `02_deployment/legacy_test_hailo_models_v2.py` |
| `test_hailo_models_3.py` | `02_deployment/legacy_test_hailo_models_v3.py` |
| `HAILO/final output/legacy_test_hailo_models.py` | `02_deployment/validate_hailo_models.py` |
| `HAILO/calibration/classifier/calib_cls_224.npy` | `03_training/calib_cls_224.npy` |
| `HAILO/calibration/lesion/calib_lesion_256.npy` | `03_training/calib_lesion_256.npy` |
| `HAILO/calibration/yolo/calib_yolo_320.npy` | `03_training/calib_yolo_320.npy` |
| `HAILO/calibration/yolo/calib_yolo_640.npy` | `03_training/calib_yolo_640.npy` |
| `HAILO/improved_scripts/export_calibration_classifier.py` | `03_training/export_calibration_classifier.py` |
| `HAILO/scripts/export_calibration_classifier.py` | `03_training/legacy_export_calibration_classifier_v1.py` |
| `HAILO/improved_scripts/export_calibration_lesion.py` | `03_training/export_calibration_lesion.py` |
| `HAILO/scripts/export_calibration_lesion.py` | `03_training/legacy_export_calibration_lesion_v1.py` |
| `HAILO/improved_scripts/export_calibration_yolo.py` | `03_training/export_calibration_yolo.py` |
| `HAILO/scripts/export_calibration_yolo.py` | `03_training/legacy_export_calibration_yolo_v1.py` |
| `Infection Classification/train.py` | `03_training/train_disease_classifier.py` |
| `Leaf Segmentation/train.py` | `03_training/train_leaf_segmentation.py` |
| `Lesion Segmentation/train.py` | `03_training/train_lesion_segmentation.py` |
| `lesion_segmentation.har` | `04_models/hef/lesion_segmentation.har` |
| `HAILO/output/lesion_segmentation.hef` | `04_models/hef/lesion_segmentation.hef` |
| `HAILO/lesion_segmentation.har` | `04_models/hef/lesion_segmentation_1.har` |
| `HAILO/output/lesion_segmentation_compiled.har` | `04_models/hef/lesion_segmentation_compiled.har` |
| `HAILO/final output/lesion_segmentation.hef` | `04_models/hef/lesion_segmentation_unet.hef` |
| `HAILO/output/lesion_segmentation_quantized.har` | `04_models/hef/lesion_segmentation_quantized.har` |
| `potato_classifier.har` | `04_models/hef/potato_classifier.har` |
| `HAILO/output/potato_classifier.hef` | `04_models/hef/potato_classifier.hef` |
| `HAILO/potato_classifier.har` | `04_models/hef/potato_classifier_1.har` |
| `HAILO/output/potato_classifier_compiled.har` | `04_models/hef/potato_classifier_compiled.har` |
| `HAILO/final output/potato_classifier.hef` | `04_models/hef/disease_classifier_mobilenet.hef` |
| `HAILO/output/potato_classifier_quantized.har` | `04_models/hef/potato_classifier_quantized.har` |
| `yolov8n_seg.har` | `04_models/hef/yolov8n_seg.har` |
| `HAILO/output/yolov8n_seg.hef` | `04_models/hef/yolov8n_seg.hef` |
| `HAILO/yolov8n_seg.har` | `04_models/hef/yolov8n_seg_1.har` |
| `HAILO/output/yolov8n_seg_compiled.har` | `04_models/hef/yolov8n_seg_compiled.har` |
| `HAILO/final output/yolov8n_seg.hef` | `04_models/hef/leaf_segmentation_yolov8n.hef` |
| `HAILO/output/yolov8n_seg_quantized.har` | `04_models/hef/yolov8n_seg_quantized.har` |
| `Leaf Segmentation/best.onnx` | `04_models/onnx/leaf_segmentation_best.onnx` |
| `HAILO/models/lesion_segmentation.onnx` | `04_models/onnx/lesion_segmentation.onnx` |
| `Lesion Segmentation/lesion_segmentation.onnx` | `04_models/onnx/legacy_lesion_segmentation_v1.onnx` |
| `HAILO/models/disease_classifier_mobilenet.onnx` | `04_models/onnx/disease_classifier_mobilenet.onnx` |
| `Infection Classification/disease_classifier_mobilenet.onnx` | `04_models/onnx/legacy_disease_classifier_v1.onnx` |
| `HAILO/models/leaf_segmentation_yolov8n.onnx` | `04_models/onnx/leaf_segmentation_yolov8n.onnx` |
| `all_files_filtered.txt` | `05_configs/all_files_filtered.txt` |
| `all_files_filtered_ascii.txt` | `05_configs/all_files_filtered_ascii.txt` |
| `manifest_summary.txt` | `05_configs/manifest_summary.txt` |
| `model_capabilities.yaml` | `05_configs/model_capabilities.yaml` |
| `edge_requirements.txt` | `05_configs/edge_requirements.txt` |
| `HAILO/improved_scripts/compile_all_to_hef.sh` | `06_scripts/compile_all_to_hef.sh` |
| `HAILO/scripts/compile_all_to_hef.sh` | `06_scripts/legacy_compile_all_to_hef_v1.sh` |
| `HAILO/scripts/measure_snr.py` | `06_scripts/measure_snr.py` |
| `scripts/search_repo.sh` | `06_scripts/search_repo.sh` |
| `scripts/setup_search.sh` | `06_scripts/setup_search.sh` |
| `t.py` | `06_scripts/experimental_tensor_test.py` |
| `scripts/validate-all.sh` | `06_scripts/validate-all.sh` |
| `scripts/validate-skills.sh` | `06_scripts/validate-skills.sh` |
| `scripts/validate-templates.sh` | `06_scripts/validate-templates.sh` |
| `scripts/validate-workflows.sh` | `06_scripts/validate-workflows.sh` |
| `scripts/verify_export.py` | `06_scripts/verify_export.py` |
| `scripts/verify_model.py` | `06_scripts/verify_model.py` |
| `scripts/verify_orchestration.py` | `06_scripts/verify_orchestration.py` |
| `scripts/verify_unet_export.py` | `06_scripts/verify_unet_export.py` |
| `scripts/verify_yolo_export.py` | `06_scripts/verify_yolo_export.py` |
| `stereo_calib/E.npy` | `07_hardware_calibration/E.npy` |
| `stereo_calib/F.npy` | `07_hardware_calibration/F.npy` |
| `stereo_calib/R.npy` | `07_hardware_calibration/R.npy` |
| `stereo_calib/T.npy` | `07_hardware_calibration/T.npy` |
| `stereo_calib/cameraMatrixL.npy` | `07_hardware_calibration/cameraMatrixL.npy` |
| `stereo_calib/cameraMatrixR.npy` | `07_hardware_calibration/cameraMatrixR.npy` |
| `HAILO/scripts/classifier_optimization.alls` | `07_hardware_calibration/classifier_optimization.alls` |
| `stereo_calib/distL.npy` | `07_hardware_calibration/distL.npy` |
| `stereo_calib/distR.npy` | `07_hardware_calibration/distR.npy` |
| `HAILO/estimation.csv` | `07_hardware_calibration/estimation.csv` |
| `krishi_eye_teensy/krishi_eye_teensy.ino` | `07_hardware_calibration/krishi_eye_teensy.ino` |
| `HAILO/scripts/lesion_optimization.alls` | `07_hardware_calibration/lesion_optimization.alls` |
| `HAILO/output/lesion_segmentation_profile.html` | `07_hardware_calibration/lesion_segmentation_profile.html` |
| `HAILO/output/potato_classifier_profile.html` | `07_hardware_calibration/potato_classifier_profile.html` |
| `stereo_calib/reprojection_error.npy` | `07_hardware_calibration/reprojection_error.npy` |
| `scripts/search_repo.ps1` | `07_hardware_calibration/search_repo.ps1` |
| `scripts/setup_search.ps1` | `07_hardware_calibration/setup_search.ps1` |
| `scripts/validate-all.ps1` | `07_hardware_calibration/validate-all.ps1` |
| `scripts/validate-skills.ps1` | `07_hardware_calibration/validate-skills.ps1` |
| `scripts/validate-templates.ps1` | `07_hardware_calibration/validate-templates.ps1` |
| `scripts/validate-workflows.ps1` | `07_hardware_calibration/validate-workflows.ps1` |
| `HAILO/scripts/yolo_compilation.alls` | `07_hardware_calibration/yolo_compilation.alls` |
| `HAILO/scripts/yolo_optimization.alls` | `07_hardware_calibration/yolo_optimization.alls` |
| `HAILO/output/yolov8n_seg_profile.html` | `07_hardware_calibration/yolov8n_seg_profile.html` |
