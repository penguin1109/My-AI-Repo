configs = {
'data_set' : {
    'recipe' : 'DB_CBL2_SE_DENOISER', 
    'aims_interface_configuration' : "/project/workSpace/gauss-nas/dev/aims/aims-recipe/app-assets/aims-core-configuration.json",
    'batch_size' : 4,
    'augmentations' : [
                     None
    ],
},

'optimizer' :{
    'type' : "Adam",
    'lr' : 1e-2,
    'wd' : 1e-4
},

'scheduler' : {
    'type' : "MultiStepLR",
    'milestones' : [30, 80],
    'gamma' : 0.5
}, 

'loss' :[
    {
      'type': "IntensityMarginLoss",
      'lambda': 1.0
    },
    {
      'type': "BlockVarianceLoss",
      'lambda': 1.0
    },
    {
      'type': "NoiseLevelLoss",
      'lambda': 1.0
    },
    {
      'type': "GradientLoss",
      'lambda': 1.0
    }],

'train_pipeline' : {
  'strategy': [
 {
    'type' : "random",
    'epoch' : 20
 },
  {
    'type' : "zero_selection",
    'epoch' : 20  
  }
  ]},

'eval_pipeline' : {'epoch' : 10},

'mlops_pipeline' :{
  'tracking_uri' : "http://aims-mlops-basic-dev.api.sddc-gl.skhynix.com",
  'experiment_name' : "multirun",
  'run_name' : "denoiser-test",
  'user_name' : "jihye"
  },
'model' :{
  'backbone' : {
    'name' : "UNet", 
    'from_scratch' : True,
    'pretrained_pt' : ""
    }
  }
}
