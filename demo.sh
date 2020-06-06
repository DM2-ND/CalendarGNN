# Default
python main.py --model calendargnn --label gender --num_epochs 10

# Run CalendarGNN for predicting income using GPU
#python main.py --model calendargnn --label income --cuda

# Run CalendarGNN-Attn for predicting age using GPU
#python main.py --model calendargnnattn --label age --cuda

# Specifying spatial/temporal unit embedding size and pattern embedding size
#python main.py --model calendargnn --hidden_dim 512 --pattern_dim 256
