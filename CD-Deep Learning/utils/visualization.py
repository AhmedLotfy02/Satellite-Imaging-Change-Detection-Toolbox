from tensorboardX import SummaryWriter

class Visualization:
    def __init__(self):
        self.writer = ''

    def create_summary(self, model_type='U_Net'):
        self.writer = SummaryWriter(comment='-' +model_type)

    def add_scalar(self, epoch, value, params='loss'):
        self.writer.add_scalar(params, value, global_step=epoch)

    def add_iamge(self, tag, img_tensor):
        
        self.writer.add_iamge(tag, img_tensor)

    def add_graph(self, model):
        self.writer.add_graph(model)

    def close_summary(self):
        self.writer.close()