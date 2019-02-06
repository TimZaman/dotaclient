{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 16,
      batch_size: 8,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.7',
      epochs: 4,
      expname: 'exp3',
      jobname: 'job3',
      learning_rate: '1e-4',
      entropy_coef: '0.02',
      max_dota_time: 600,
      optimizers: 1,
      pretrained_model: 'exp3/job2/model_000006414.pt',
      rollout_size: 9999,
      seq_len: 256,
      seq_per_epoch: 32,
    },
  },
}