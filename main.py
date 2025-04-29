import openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

plot_graphs = True

# Soma dos RMS
numbers = [
    96203,
    96322,
    93323,
    95497,
    93492,
    93417,
    94049,
]

def sum_dig(number):
    return sum(int(dig) for dig in str(number))

sum_grup = [sum_dig(number) for number in numbers]

for number, soma in zip(numbers, sum_grup):
    print(f"soma de digitos de {number}: {soma}")

sum_total = sum(sum_grup)
print(f"soma total: {sum_total}")

# Puxa o dataset
dataset_id = sum_total
dataset = openml.datasets.get_dataset(dataset_id)

print(dataset)

X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

print("Shape das entradas:", X.shape)
print("Primeiras entradas:")
print(X.head())
print("Primeiros rótulos:")
print(y.head())

# Pré-processamento
X = X.iloc[4:, :8]
y = y.iloc[4:]

for col in X.columns:
    if X[col].dtype == object or X[col].dtype.name == "category":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

if y.dtype == object or y.dtype.name == "category":
    le = LabelEncoder()
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# ----- MLP Sklearn -----
mlp = MLPClassifier( 
    hidden_layer_sizes=(173,), 
    activation="relu", 
    solver="adam", 
    max_iter=1000, 
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"precisão (sklearn) no conjunto de teste: {acc:.4f}")

# ----- MLP FEITA NA MÃO (NumPy) -----

# Funções de ativação
def relu(X):
    return np.maximum(0, X)

def relu_derivation(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def sigmoid_derivation(x):
    s = sigmoid(x)
    return s * (1 - s)

# Transformar para NumPy
X_train_np = X_train.values
y_train_np = y_train.reshape(-1, 1)

X_test_np = X_test.values
y_test_np = y_test.reshape(-1, 1)

# Arquitetura
input_size = X_train_np.shape[1]
hidden_size = 173
output_size = 1

# Inicialização dos pesos
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 1000

loss_history = []
acc_history = []

# Treinamento
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X_train_np, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss
    loss = -np.mean(y_train_np * np.log(a2 + 1e-8) + (1 - y_train_np) * np.log(1 - a2 + 1e-8))
    loss_history.append(loss)

    y_pred_train = (a2 > 0.5).astype(int)
    acc_train = np.mean(y_pred_train == y_train_np)
    acc_history.append(acc_train)

    # Backprop
    dz2 = a2 - y_train_np
    dw2 = np.dot(a1.T, dz2) / X_train_np.shape[0]
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * relu_derivation(z1)
    dw1 = np.dot(X_train_np.T, dz1) / X_train_np.shape[0]
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Atualização dos pesos
    W1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

# ----- Avaliação -----

# Forward no conjunto de teste!
z1_test = np.dot(X_test_np, W1) + b1
a1_test = relu(z1_test)

z2_test = np.dot(a1_test, W2) + b2
a2_test = sigmoid(z2_test)

y_pred_base = (a2_test > 0.5).astype(int)

acc_np = np.mean(y_pred_base == y_test_np)
print(f"precisão (numpy) no conjunto de teste: {acc_np:.4f}")

# ------------------- Comparação final -------------------

print("\n------ Comparação de Modelos ------")
print(f"Sklearn MLPClassifier - Precisão: {acc:.4f}")
print(f"Numpy MLP Manual      - Precisão: {acc_np:.4f}")

diff = abs(acc - acc_np)
print(f"\nDiferença de precisão: {diff:.4f}")

if diff <= 0.05:
    print("✅ As precisões são bem próximas! Sua rede manual está funcionando corretamente!")
else:
    print("⚠️  Há uma diferença considerável. Pode ser por regularizações, inicialização ou método de otimização diferentes.")

# ------------------- Gráficos opcionais -------------------

fig, ax1 = plt.subplots(figsize=(10, 5))

epochs_range = range(1, epochs + 1)

color = 'tab:red'
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs_range, loss_history, color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # cria um segundo eixo y
color = 'tab:blue'
ax2.set_ylabel('Acurácia', color=color)
ax2.plot(epochs_range, acc_history, color=color, label='Acurácia')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Loss e Acurácia durante o Treinamento')
fig.tight_layout()

plt.figure(figsize=(8, 5))
plt.bar(["Sklearn MLP", "NumPy MLP"], [acc, acc_np], color=["green", "orange"])
plt.title("Comparação de Precisão entre os Modelos")
plt.ylabel("Precisão")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()

# Exibindo a precisão no terminal
print(f"Precisão do MLP (sklearn): {acc:.4f}")
print(f"Precisão do MLP (NumPy): {acc_np:.4f}")
