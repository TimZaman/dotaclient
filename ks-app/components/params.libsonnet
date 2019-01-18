{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 4,
      batch_size: 4,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.4',
      expname: 'exp1',
      jobname: 'job6-ppo',
      learning_rate: '1e-5',
      max_dota_time: 210,
      optimizers: 1,
      pretrained_model: 'a/20180115_160800/model_000007163.pt',
      rollout_size: 9999,
    },
  },
}