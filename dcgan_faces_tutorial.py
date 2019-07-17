
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import sys

from models.constants import *
from models.generator32 import *
from models.discriminator32 import *





######################################################################
# Implementation
# --------------
#
# With our input parameters set and the dataset prepared, we can now get
# into the implementation. We will start with the weigth initialization
# strategy, then talk about the generator, discriminator, loss functions,
# and training loop in detail.
#
# Weight Initialization
# ~~~~~~~~~~~~~~~~~~~~~
#
# From the DCGAN paper, the authors specify that all model weights shall
# be randomly initialized from a Normal distribution with mean=0,
# stdev=0.2. The ``weights_init`` function takes an initialized model as
# input and reinitializes all convolutional, convolutional-transpose, and
# batch normalization layers to meet this criteria. This function is
# applied to the models immediately after initialization.
#

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)





######################################################################
# Training
# ~~~~~~~~
#
# Finally, now that we have all of the parts of the GAN framework defined,
# we can train it. Be mindful that training GANs is somewhat of an art
# form, as incorrect hyperparameter settings lead to mode collapse with
# little explanation of what went wrong. Here, we will closely follow
# Algorithm 1 from Goodfellow’s paper, while abiding by some of the best
# practices shown in `ganhacks <https://github.com/soumith/ganhacks>`__.
# Namely, we will “construct different mini-batches for real and fake”
# images, and also adjust G’s objective function to maximize
# :math:`logD(G(z))`. Training is split up into two main parts. Part 1
# updates the Discriminator and Part 2 updates the Generator.
#
# **Part 1 - Train the Discriminator**
#
# Recall, the goal of training the discriminator is to maximize the
# probability of correctly classifying a given input as real or fake. In
# terms of Goodfellow, we wish to “update the discriminator by ascending
# its stochastic gradient”. Practically, we want to maximize
# :math:`log(D(x)) + log(1-D(G(z)))`. Due to the separate mini-batch
# suggestion from ganhacks, we will calculate this in two steps. First, we
# will construct a batch of real samples from the training set, forward
# pass through :math:`D`, calculate the loss (:math:`log(D(x))`), then
# calculate the gradients in a backward pass. Secondly, we will construct
# a batch of fake samples with the current generator, forward pass this
# batch through :math:`D`, calculate the loss (:math:`log(1-D(G(z)))`),
# and *accumulate* the gradients with a backward pass. Now, with the
# gradients accumulated from both the all-real and all-fake batches, we
# call a step of the Discriminator’s optimizer.
#
# **Part 2 - Train the Generator**
#
# As stated in the original paper, we want to train the Generator by
# minimizing :math:`log(1-D(G(z)))` in an effort to generate better fakes.
# As mentioned, this was shown by Goodfellow to not provide sufficient
# gradients, especially early in the learning process. As a fix, we
# instead wish to maximize :math:`log(D(G(z)))`. In the code we accomplish
# this by: classifying the Generator output from Part 1 with the
# Discriminator, computing G’s loss *using real labels as GT*, computing
# G’s gradients in a backward pass, and finally updating G’s parameters
# with an optimizer step. It may seem counter-intuitive to use the real
# labels as GT labels for the loss function, but this allows us to use the
# :math:`log(x)` part of the BCELoss (rather than the :math:`log(1-x)`
# part) which is exactly what we want.
#
# Finally, we will do some statistic reporting and at the end of each
# epoch we will push our fixed_noise batch through the generator to
# visually track the progress of G’s training. The training statistics
# reported are:
#
# -  **Loss_D** - discriminator loss calculated as the sum of losses for
#    the all real and all fake batches (:math:`log(D(x)) + log(D(G(z)))`).
# -  **Loss_G** - generator loss calculated as :math:`log(D(G(z)))`
# -  **D(x)** - the average output (across the batch) of the discriminator
#    for the all real batch. This should start close to 1 then
#    theoretically converge to 0.5 when G gets better. Think about why
#    this is.
# -  **D(G(z))** - average discriminator outputs for the all fake batch.
#    The first number is before D is updated and the second number is
#    after D is updated. These numbers should start near 0 and converge to
#    0.5 as G gets better. Think about why this is.
#
# **Note:** This step might take a while, depending on how many epochs you
# run and if you removed some data from the dataset.
#

# Training Loop
def train(netG, netD, optimizerG, optimizerD, dataloader, device, fixed_noise):
    # Lists to keep track of progress

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    checkpoint = None
    resume_epoch=0
    if len(sys.argv) > 1:
        checkpoint = torch.load(sys.argv[1])
        netG.load_state_dict(checkpoint['generator_state'])
        netD.load_state_dict(checkpoint['discriminator_state'])
        resume_epoch = checkpoint['epoch']+1
        print("Resuming from epoch " + str(resume_epoch))

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(resume_epoch, num_epochs):

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        #save model after each epoch
        save_checkpoint({
            'generator_state':netG.state_dict(),
            'discriminator_state':netD.state_dict(),
            'epoch':epoch
        })

    return img_list, G_losses, D_losses

######################################################################
# Results
# -------
#
# Finally, lets check out how we did. Here, we will look at three
# different results. First, we will see how D and G’s losses changed
# during training. Second, we will visualize G’s output on the fixed_noise
# batch for every epoch. And third, we will look at a batch of real data
# next to a batch of fake data from G.
#
# **Loss versus training iteration**
#
# Below is a plot of D & G’s losses versus training iterations.
#
def show_results(img_list, G_losses, D_losses):

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    ######################################################################
    # **Visualization of G’s progression**
    #
    # Remember how we saved the generator’s output on the fixed_noise batch
    # after every epoch of training. Now, we can visualize the training
    # progression of G with an animation. Press the play button to start the
    # animation.
    #

    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())


    ######################################################################
    # **Real Images vs. Fake Images**
    #
    # Finally, lets take a look at some real images and fake images side by
    # side.
    #

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()


if __name__ == '__main__':
    # Set random seem for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    ######################################################################
    # Data
    # ----
    #
    # In this tutorial we will use the `Celeb-A Faces
    # dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ which can
    # be downloaded at the linked site, or in `Google
    # Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__.
    # The dataset will download as a file named *img_align_celeba.zip*. Once
    # downloaded, create a directory named *celeba* and extract the zip file
    # into that directory. Then, set the *dataroot* input for this notebook to
    # the *celeba* directory you just created. The resulting directory
    # structure should be:
    #
    # ::
    #
    #    /path/to/celeba
    #        -> img_align_celeba
    #            -> 188242.jpg
    #            -> 173822.jpg
    #            -> 284702.jpg
    #            -> 537394.jpg
    #               ...
    #
    # This is an important step because we will be using the ImageFolder
    # dataset class, which requires there to be subdirectories in the
    # dataset’s root folder. Now, we can create the dataset, create the
    # dataloader, set the device to run on, and finally visualize some of the
    # training data.
    #

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


    ######################################################################
    # Now, we can instantiate the generator and apply the ``weights_init``
    # function. Check out the printed model to see how the generator object is
    # structured.
    #

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        print("Using " + str(ngpu) + " GPU(s)")

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Now, as with the generator, we can create the discriminator, apply the
    # ``weights_init`` function, and print the model’s structure.
    #

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    ######################################################################
    # Loss Functions and Optimizers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # With :math:`D` and :math:`G` setup, we can specify how they learn
    # through the loss functions and optimizers. We will use the Binary Cross
    # Entropy loss
    # (`BCELoss <https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss>`__)
    # function which is defined in PyTorch as:
    #
    # .. math:: \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
    #
    # Notice how this function provides the calculation of both log components
    # in the objective function (i.e. :math:`log(D(x))` and
    # :math:`log(1-D(G(z)))`). We can specify what part of the BCE equation to
    # use with the :math:`y` input. This is accomplished in the training loop
    # which is coming up soon, but it is important to understand how we can
    # choose which component we wish to calculate just by changing :math:`y`
    # (i.e. GT labels).
    #
    # Next, we define our real label as 1 and the fake label as 0. These
    # labels will be used when calculating the losses of :math:`D` and
    # :math:`G`, and this is also the convention used in the original GAN
    # paper. Finally, we set up two separate optimizers, one for :math:`D` and
    # one for :math:`G`. As specified in the DCGAN paper, both are Adam
    # optimizers with learning rate 0.0002 and Beta1 = 0.5. For keeping track
    # of the generator’s learning progression, we will generate a fixed batch
    # of latent vectors that are drawn from a Gaussian distribution
    # (i.e. fixed_noise) . In the training loop, we will periodically input
    # this fixed_noise into :math:`G`, and over the iterations we will see
    # images form out of the noise.
    #

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)



    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))




    img_list, G_losses, D_losses = train(netG, netD, optimizerG, optimizerD, dataloader, device, fixed_noise)#this is really trash so i should get around to making it better eventually
    show_results(img_list, G_losses, D_losses)

######################################################################
# Where to Go Next
# ----------------
#
# We have reached the end of our journey, but there are several places you
# could go from here. You could:
#
# -  Train for longer to see how good the results get
# -  Modify this model to take a different dataset and possibly change the
#    size of the images and the model architecture
# -  Check out some other cool GAN projects
#    `here <https://github.com/nashory/gans-awesome-applications>`__
# -  Create GANs that generate
#    `music <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>`__
#
