import torch
from models import XVLMBase, load_pretrained, build_mlp_cosmos
import torch.nn.functional as F
from models.xbert import BertConfig
from models.xroberta import RobertaConfig

class XVLM(XVLMBase):
    def __init__(self, config):
        config_text = RobertaConfig.from_json_file(config['text_config']) \
            if config['use_roberta'] else BertConfig.from_json_file(config['text_config'])
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=config_text
                         )

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.cls_head = build_mlp_cosmos(input_dim=self.text_width * 2, output_dim=2)
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]
        # self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids_1, text_atts_1, text_ids_2, text_atts_2,targets,train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)

        # text_embeds = self.get_text_embeds(text_ids_1, text_atts_1)
        # text_embeds2 = self.get_text_embeds(text_ids_1, text_atts_1)
        
        output_cls_1 = self.get_cross_embeds(
            image_embeds,image_atts,
            text_ids=text_ids_1, 
            text_atts=text_atts_1
        )[:, 0, :]

        output_cls_2 = self.get_cross_embeds(
            image_embeds,image_atts,
            text_ids=text_ids_2, 
            text_atts=text_atts_2
        )[:, 0, :]

        output_cls = torch.cat([output_cls_1,output_cls_2], dim=-1)
        prediction = self.cls_head(output_cls)
        
        return F.cross_entropy(prediction, targets) if train else prediction