{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 5,
      epochs: 4,
      seq_per_epoch: 16,
      batch_size: 4,
      seq_len: 256,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.5',
      expname: 'exp1',
      jobname: 'job15-ppo',
      learning_rate: '1e-5',
      max_dota_time: 420,
      optimizers: 1,
      pretrained_model: 'exp1/job14-ppo/model_000002085.pt',
      rollout_size: 9999,
    },
  },
}