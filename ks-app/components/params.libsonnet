{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 5,
      batch_size: 4,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.5',
      expname: 'exp1',
      jobname: 'job14-ppo',
      learning_rate: '2e-6',
      max_dota_time: 300,
      optimizers: 1,
      pretrained_model: 'exp1/job13-ppo/model_000001385.pt',
      rollout_size: 9999,
    },
  },
}