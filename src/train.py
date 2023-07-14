import torch

from model import CycleGAN

def train_cyclegan():
  # Loss parameters
  cyc_tradeoff_parameter=10
  identity_tradeoff_parameter=5

  def training_iteration(x,
                        y,
                        real_labels,
                        fake_labels,
                        optimizer,
                        criterion, Gx_y, Gy_x, Dx, Dy):

    # ======================================================== #
    #                      TRAIN DISCRIMINATORS                #
    # ======================================================== #

    def adversarial_loss(G, D, source, target):

      # Discriminator should learn to classify real and fake features
      # correctly. Thus we give it a set for real features with real labels
      # and a set of fake features with fake labels.
      real_outputs = D.forward(target.unsqueeze(0))
      d_loss_real = criterion(real_outputs, real_labels)
      real_score = real_outputs

      # Generate fake features
      # from IPython.core.debugger import Tracer; Tracer()() 
      z = source
      fake_features = G.forward(z)
      fake_outputs = D.forward(fake_features.unsqueeze(0))
      d_loss_fake = criterion(fake_outputs, fake_labels)
      fake_score = fake_outputs

      d_loss = d_loss_real + d_loss_fake
      return d_loss, real_score, fake_score

    dy_loss, real_score, fake_score = adversarial_loss(G=Gx_y, D=Dy, source=x, target=y)
    dx_loss, real_score, fake_score = adversarial_loss(G=Gy_x, D=Dx, source=y, target=x)


    # ======================================================== #
    #                      TRAIN GENERATORS                    #
    # ======================================================== #

    def cycle_consistency_loss(Gx_y, Gy_x, x, y):
      forward_loss = torch.norm(Gy_x.forward(Gx_y.forward(x)) - x)
      inverse_forward_loss = torch.norm(Gx_y.forward(Gy_x.forward(y)) - y)
      return (forward_loss + inverse_forward_loss)
    
    def identity_loss(Gx_y, Gy_x, x, y):
      return torch.norm(Gx_y.forward(y) - y) + torch.norm(Gy_x.forward(x) - x)
    
    
    # Train Gx_y and Gy_x
    full_loss = dx_loss + dy_loss + cycle_consistency_loss(Gx_y, Gy_x, x, y) * cyc_tradeoff_parameter + \
      identity_loss(Gx_y, Gy_x, x, y) * identity_tradeoff_parameter

    optimizer.zero_grad()
    optimizer.zero_grad()
    full_loss.backward()
    optimizer.step()

    # Return all losses
    return full_loss

  device = 'cpu'

  batch_size=1
  cycleGAN = CycleGAN()
  cycleGAN.to(device)

  # Create labels which are later used as input for the BCU loss
  real_labels = torch.ones(batch_size, 1)
  fake_labels = torch.zeros(batch_size, 1)

  optimizer=torch.optim.Adam(cycleGAN.parameters(), lr=0.0002)
  criterion=torch.nn.BCELoss()
  source_features = torch.randn(1, 24, 128).to(device)
  target_features = torch.randn(1, 24, 128).to(device)

  def train(num_epochs):
    epochs = []
    losses = []

    for epoch in range(num_epochs):
      if epoch % 10 == 0:
        print(f'Epoch {epoch}')
      loss = training_iteration(
        source_features,
        target_features,
        optimizer,
        criterion,
        cycleGAN.Gx_y,
        cycleGAN.Gy_x,
        cycleGAN.Dx,
        cycleGAN.Dy,
      )
      epochs.append(epoch)
      losses.append(loss.item())

    import matplotlib.pyplot as plt
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('dx_loss')
    plt.show()

  train(num_epochs = 3)