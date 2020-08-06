from __future__ import print_function

def create_model(opt):
    if opt["model"] == 'sr_resnet':
        from .sr_resnet_model import SRResNetModel
        model = SRResNetModel(opt)

    # elif opt["model"] == 'sr_resnet_test':
    #     from .sr_resnet_test_model import SRResNetTestModel
    #     model = SRResNetTestModel()
    #
    # elif opt["model"] == 'sr_gan':
    #     from .sr_gan_model import SRGANModel
    #     model = SRGANModel()

    elif opt["model"] == "espcn":
        from .espcn import ESPCNModel
        model = ESPCNModel(opt)

    elif opt["model"] == "sft_resnet":
        from .sft_resnet_model import SFTResNetModel
        model = SFTResNetModel(opt)
    
    elif opt["model"] == "espcn_3D":
        from .espcn3D import ESPCN3DModel
        model = ESPCN3DModel(opt)
        
    else:
        raise NotImplementedError('Model [%s] not recognized.' % opt["model"])

    print('Model [%s] is created.' % model.name())
    return model
