from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

class Vehicle_Classifier:
	def __init__(self, car_features, not_car_features):
		self.svc = LinearSVC()
		self.car_features = car_features
		self.not_car_features = not_car_features
		self.X_train = []
		self.X_test = []
		self.y_train = []
		self.y_test = []
		self.X_scaler = None

	def create_array_stack_features(self):
		X = np.vstack((self.car_features, self.not_car_features)).astype(np.float64)
		return X

	def create_labels_vector(self):
		y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.not_car_features))))
		return y

	def split_data(self):
		X = self.create_array_stack_features()
		y = self.create_labels_vector()

		rand_state = np.random.randint(0, 100)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

	def set_X_scaler(self):
		self.X_scaler = StandardScaler().fit(self.X_train)

	def per_column_scaler(self):
		self.set_X_scaler()

		self.X_train = self.X_scaler.transform(self.X_train)
		self.X_test = self.X_scaler.transform(self.X_test)

	def fit(self):
		self.split_data()
		self.per_column_scaler()

		print('Feature vector length:', len(self.X_train[0]))
		t = time.time()
		self.svc.fit(self.X_train, self.y_train)
		t2 = time.time()

		print(round(t2-t, 2), 'Seconds to train SVC...')
		# Check the score of the SVC
		print('Test Accuracy of SVC = ', round(self.svc.score(self.X_test, self.y_test), 4))
		# Check the prediction time for a single sample
		t=time.time()
