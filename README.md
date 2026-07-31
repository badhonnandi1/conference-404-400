# Compression-Resilient Cryptographic Authentication for Secure Surveillance Video Integrity


## Project Structure

```text
video-authentication/
├── configs/
│   └── default.yaml
├── data/
│   ├── originals/
│   ├── segments/
│   ├── sampled_frames/
│   ├── metadata/
│   └── manifests/
├── logs/
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── authentication/
│   │   ├── __init__.py
│   │   ├── auth_record_storage.py
│   │   ├── canonicalization.py
│   │   ├── digest.py
│   │   ├── digest_storage.py
│   │   ├── hmac_auth.py
│   │   └── quantization.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── aggregation.py
│   │   ├── alignment.py
│   │   ├── device.py
│   │   ├── feature_storage.py
│   │   ├── fusion.py
│   │   ├── normalization.py
│   │   ├── normalization_storage.py
│   │   ├── resnet_features.py
│   │   ├── temporal_features.py
│   │   ├── temporal_sampling.py
│   │   └── temporal_storage.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_utils.py
│   │   └── ffmpeg_utils.py
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── comparison.py
│   │   ├── comparison_storage.py
│   │   ├── hamming.py
│   │   └── segment_alignment.py
│   └── video/
│       ├── __init__.py
│       ├── metadata.py
│       ├── segmentation.py
│       └── frame_sampling.py
├── tests/
│   ├── __init__.py
│   ├── test_digest.py
│   ├── test_digest_storage.py
│   ├── test_feature_aggregation.py
│   ├── test_feature_alignment.py
│   ├── test_feature_fusion.py
│   ├── test_hamming.py
│   ├── test_feature_normalization.py
│   ├── test_feature_storage.py
│   ├── test_metadata.py
│   ├── test_normalization_storage.py
│   ├── test_quantization.py
│   ├── test_repository_hygiene.py
│   ├── test_resnet_features.py
│   ├── test_segmentation.py
│   ├── test_segment_alignment.py
│   ├── test_comparison.py
│   ├── test_comparison_storage.py
│   ├── test_temporal_features.py
│   ├── test_temporal_sampling.py
│   ├── test_temporal_storage.py
│   └── test_frame_sampling.py
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```
