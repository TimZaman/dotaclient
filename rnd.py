
class Rnd(nn.Module):

    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 64)
        self.affine2 = nn.Linear(64, 64)
        self.affine3 = nn.Linear(64, 64)

    def forward(self, *args, **kwargs):
        inputs = list(args) + list(kwargs.values())
        inputs = inputs[:2]  # TODO(tzaman): fix me being lazy
        inputs = torch.cat([torch.flatten(inp) for inp in inputs[:2] if inp is not []])
        x = self.affine1(inputs) # (32,)
        x = F.relu(self.affine2(x)) # (32,)
        x = F.relu(self.affine3(x)) # (32,)
        return x

rnd_fixed = Rnd()
rnd_fixed.requires_grad = False
rnd = Rnd()
rnd_lr = 1e-3
rnd_optimizer = optim.Adam(rnd.parameters(), lr=rnd_lr)


        # of = rnd_fixed(location_state, env_state, enemy_nonheroes, allied_nonheroes)
        # o = rnd(location_state, env_state, enemy_nonheroes, allied_nonheroes)
        # rnd_optimizer.zero_grad()
        # rnd_loss = torch.nn.functional.mse_loss(of, o)
        # rnd_loss.backward()
        # rnd_optimizer.step()
        rnd_loss = 0
        logging.info('rnd loss=', rnd_loss)
