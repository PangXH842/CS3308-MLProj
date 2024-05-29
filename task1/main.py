from obtain_aig import obtain_aig
from evaluate_aig import evaluate_aig
from example import generate_data

state = 'alu2_0130622'

print("Obtaining AIG")
obtain_aig(state)

print("Evaluating AIG")
eval = evaluate_aig(state)
print(eval)

print("Generating AIG data")
data = generate_data(state)
print(data)