class Opt:
	def __init__(self):
		self.dataset = "furnitureStyle"
		self.dataset_path = "./dataset/Bonn_Furniture_Styles_Dataset"
		self.batch_size = 32
		self.embed_dim = 512
		self.hashing = False
		self.retrieve_by_random = True