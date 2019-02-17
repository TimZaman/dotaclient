{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 32,
      batch_size: 8,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.8',
      entropy_coef: '0.0',
      epochs: 1,
      expname: 'exp6',
      jobname: 'job3',
      learning_rate: '2e-6',
      max_dota_time: 600,
      optimizers: 1,
      pretrained_model: 'exp6/job3/model_000002860.pt',
      rollout_size: 9999,
      seq_len: 1400,
      seq_per_epoch: 16,
      vf_coef: '0.0',
    },
  },
}