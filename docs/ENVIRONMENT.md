# 环境配置说明

- 安装 PyTorch（建议 CUDA 版本）：参考 https://pytorch.org/get-started/locally/
- 在本仓库根目录执行：`pip install -e .` 安装 fair-esm 依赖
- 额外依赖：`pip install pytest tqdm`
- 可选：使用 GPU 运行以加速 ESM 编码和训练

运行示例：
- 训练：`PYTHONPATH=. python scripts/train_ppi.py --data_dir kaggle_dataset --esm_model esm2_t6_8M_UR50D --freeze_esm --epochs 1 --batch_size 4 --precision auto --device cuda:0 --output_dir outputs`
- 推理（单对）：`PYTHONPATH=. python scripts/infer_ppi.py --esm_model esm2_t6_8M_UR50D --seq_a <SEQ_A> --seq_b <SEQ_B> --threshold 0.5`
- 推理（批量CSV）：`PYTHONPATH=. python scripts/infer_ppi.py --esm_model esm2_t6_8M_UR50D --test_csv kaggle_dataset/test.csv --out_csv submission.csv --binary --precision auto --device cuda:0`
- 使用训练权重推理：`PYTHONPATH=. python scripts/infer_ppi.py --esm_model esm2_t6_8M_UR50D --test_csv kaggle_dataset/test.csv --out_csv submission.csv --precision auto --device cuda:0 --checkpoint outputs/se_cai_last.pt`

蒸馏训练（Teacher: 冻结 ESM-2 + checkpoint，Student: CNN）：
- `PYTHONPATH=. python scripts/train_distill.py --data_dir kaggle_dataset --teacher_esm_model esm2_t33_650M_UR50D --teacher_checkpoint outputs/se_cai_last.pt --epochs 1 --batch_size 8 --precision auto --device cuda:0 --output_dir outputs`

学生模型推理：
- 单对：`PYTHONPATH=. python scripts/infer_student.py --seq_a <SEQ_A> --seq_b <SEQ_B> --checkpoint outputs/student_last.pt --precision auto --device cuda:0`
- 批量：`PYTHONPATH=. python scripts/infer_student.py --test_csv kaggle_dataset/test.csv --out_csv student_submission.csv --checkpoint outputs/student_last.pt --precision auto --device cuda:0`

双流学生蒸馏训练（PPI-D²Feat）：
- `PYTHONPATH=. python scripts/train_d2feat.py --data_dir kaggle_dataset --teacher_esm_model esm2_t33_650M_UR50D --teacher_checkpoint outputs/se_cai_last.pt --epochs 1 --batch_size 8 --precision auto --device cuda:0 --output_dir outputs`

双流学生推理：
- 单对：`PYTHONPATH=. python scripts/infer_d2feat_student.py --seq_a <SEQ_A> --seq_b <SEQ_B> --checkpoint outputs/d2feat_student_last.pt --precision auto --device cuda:0 --esm_dim 1280`
- 批量：`PYTHONPATH=. python scripts/infer_d2feat_student.py --test_csv kaggle_dataset/test.csv --out_csv d2feat_submission.csv --checkpoint outputs/d2feat_student_last.pt --precision auto --device cuda:0 --esm_dim 1280`
输出：
- 训练会在 `outputs/val_metrics.csv` 写入每个 epoch 的验证指标；在 `outputs/val_predictions_epoch_<N>.csv` 写入验证集每个样本的概率与预测。默认保存模型权重到 `outputs/se_cai_last.pt`。
