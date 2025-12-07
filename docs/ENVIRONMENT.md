# 环境配置说明

- 安装 PyTorch（建议 CUDA 版本）：参考 https://pytorch.org/get-started/locally/
- 在本仓库根目录执行：`pip install -e .` 安装 fair-esm 依赖
- 额外依赖：`pip install pytest tqdm`
- 可选：使用 GPU 运行以加速 ESM 编码和训练

运行示例：
- 训练：`PYTHONPATH=. python scripts/train_ppi.py --data_dir kaggle_dataset --esm_model esm2_t6_8M_UR50D --freeze_esm --epochs 1 --batch_size 4 --precision auto --device cuda:0`
- 推理（单对）：`PYTHONPATH=. python scripts/infer_ppi.py --esm_model esm2_t6_8M_UR50D --seq_a <SEQ_A> --seq_b <SEQ_B> --threshold 0.5`
- 推理（批量CSV）：`PYTHONPATH=. python scripts/infer_ppi.py --esm_model esm2_t6_8M_UR50D --test_csv kaggle_dataset/test.csv --out_csv submission.csv --binary --precision auto --device cuda:0`
