class Opt:
	def __init__(self):
		self.dataset = "fashion200k"
		self.dataset_path = "./dataset/Fashion200k"
		self.batch_size = 32
		self.embed_dim = 512
		self.hashing = True
		self.retrieve_by_random = False