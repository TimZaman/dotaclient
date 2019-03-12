{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 50,
      batch_size: 22,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.8',
      entropy_coef: '8e-4',
      epochs: 1,
      expname: 'exp13',
      jobname: 'job8',
      learning_rate: '2e-5',
      max_dota_time: 600,
      optimizers: 1,
      pretrained_model: 'exp13/job7/model_000001735.pt',
      rollout_size: 9999,
      seq_len: 1400,
      seq_per_epoch: 22,
      use_latest_weights_prob: '1.0',
      vf_coef: '0.00',
    },
  },
}