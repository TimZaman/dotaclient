{
  global: {
  },
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 32,
      batch_size: 16,
      expname: 'exp1',
      jobname: 'job1',
      learning_rate: 1e-4,
      max_dota_time: 210,
      optimizers: 2,
      pretrained_model: '',
      rollout_size: 9999,
      dotaclient_image_tag: 'latesst',
      dotaservice_image_tag: '0.3.3',
    },
  },
}