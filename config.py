
class Config(object):
    def __init__(self) -> None:
        self.log_path = "./logs/"
        self.gpu_id = "1"
        self.image_path = "../../datasets/UDIAT/image/"
        self.mask_path = "../../datasets/UDIAT/mask/"
        self.train_mapping_path = "../../datasets/UDIAT/train_mapping.txt"
        self.test_mapping_path = "../../datasets/UDIAT/test_mapping.txt"
        self.cut_image_path = "./cut_results/cut_image/"
        self.cut_pre_mask_path = "./cut_results/cut_pre_mask/"
        self.cut_mask_path = "./cut_results/cut_mask/"


        self.model_state_path = "./model_state/"
        self.localizer_state_path = "./model_state/"

        self.class_num = 2
        self.network_input_size = (224, 224)
        self.batch_size = 32
        self.num_workers = 32
        self.localizer_learning_rate = 0.001
        self.LOCALIZER_EPOCH = 200
        self.learning_rate = 0.001
        self.EPOCH = 120