import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.initial_dim = 2 * self.hidden_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(self.input_dim, self.initial_dim)
        # Use track_running_stats=False and replace BatchNorm with LayerNorm to handle single samples
        self.ln1 = nn.LayerNorm(self.initial_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(self.initial_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Mu and logvar projections
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
    
    def forward(self, x):
        # First encoding layer
        h1 = self.fc1(x)
        h1 = self.ln1(h1)  # LayerNorm instead of BatchNorm
        h1 = F.relu(h1)
        h1 = self.dropout1(h1)
        
        # Second encoding layer
        h2 = self.fc2(h1)
        h2 = self.ln2(h2)  # LayerNorm instead of BatchNorm
        h2 = F.relu(h2)
        h2 = self.dropout2(h2)
        
        return self.fc21(h2), self.fc22(h2)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.initial_dim = 2 * self.hidden_dim
        
        # Decoder layers
        self.fc4 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.ln4 = nn.LayerNorm(self.hidden_dim)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc5 = nn.Linear(self.hidden_dim, self.initial_dim)
        self.ln5 = nn.LayerNorm(self.initial_dim)
        self.dropout5 = nn.Dropout(dropout_rate)
        
        self.fc7 = nn.Linear(self.initial_dim, self.output_dim)
        self.output_scale = nn.Parameter(torch.tensor(2.0), requires_grad=True)
    
    def forward(self, z):
        # First decoding layer
        h4 = self.fc4(z)
        h4 = self.ln4(h4)  # LayerNorm instead of BatchNorm
        h4 = F.relu(h4)
        h4 = self.dropout4(h4)
        
        # Second decoding layer
        h5 = self.fc5(h4)
        h5 = self.ln5(h5)  # LayerNorm instead of BatchNorm
        h5 = F.relu(h5)
        h5 = self.dropout5(h5)
        
        return self.output_scale * torch.tanh(self.fc7(h5))


class HybridMultiEncoderVAE(nn.Module):
    def __init__(self, input_dims, output_dims, latent_dim, hidden_dim, nn_output_dim, dropout_rate=0.3,
                only_nn_encoder=False, only_fmri_encoders=False,
                use_nn_decoder=True, use_fmri_decoders=True):
        super(HybridMultiEncoderVAE, self).__init__()
        
        self.only_nn_encoder = only_nn_encoder
        self.only_fmri_encoders = only_fmri_encoders
        self.use_nn_decoder = use_nn_decoder
        self.use_fmri_decoders = use_fmri_decoders
        
        # Store important indices
        self.nn_encoder_idx = -1  # Initialize as -1 (not present)
        self.num_fmri_encoders = 0  # Initialize count
        
        # Determine which encoders to create based on flags
        if only_nn_encoder:
            # Only use NN encoder (last in input_dims)
            print("Using only NN encoder")
            print("with dims:", input_dims[-1])
            self.encoders = nn.ModuleList([
                Encoder(input_dims[-1], latent_dim, hidden_dim, dropout_rate)
            ])
            self.nn_encoder_idx = 0  # NN encoder is the only one at index 0
            self.num_fmri_encoders = 0
        elif only_fmri_encoders:
            # Only use fMRI encoders (exclude last dimension which is NN)
            print("Using only fMRI encoders")
            print("with dims:", input_dims)
            self.encoders = nn.ModuleList([
                Encoder(input_dim, latent_dim, hidden_dim, dropout_rate) 
                for input_dim in input_dims
            ])
            self.nn_encoder_idx = -1  # No NN encoder
            self.num_fmri_encoders = len(input_dims)
        else:
            # Use all encoders
            print("Using all encoders (fMRI + NN)")
            print("with dims:", input_dims)
            self.encoders = nn.ModuleList([
                Encoder(input_dim, latent_dim, hidden_dim, dropout_rate) 
                for input_dim in input_dims
            ])
            self.nn_encoder_idx = len(input_dims) - 1  # NN encoder is last
            self.num_fmri_encoders = len(input_dims) - 1
        
        # NN decoder (if enabled)
        self.nn_decoder = None
        if use_nn_decoder:
            print("Creating NN decoder")
            print("with dims:", latent_dim, hidden_dim, nn_output_dim)
            self.nn_decoder = Decoder(latent_dim, hidden_dim, nn_output_dim, dropout_rate)
        
        # fMRI decoders (if enabled)
        self.fmri_decoders = None
        if use_fmri_decoders:
            print("Creating fMRI decoders")
            # Use all fMRI dimensions (excluding last which is NN)
            fmri_dims = output_dims
            print("with dims:", fmri_dims)
            self.fmri_decoders = nn.ModuleList([
                Decoder(latent_dim, hidden_dim, fmri_dim, dropout_rate) 
                for fmri_dim in fmri_dims
            ])
        else:
            print("No fMRI decoders")
        

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick that handles 2D or 3D tensors
        
        If mu is 3D [batch, latent_dim, num_encoders], returns 3D z
        If mu is 2D [batch, latent_dim], returns 2D z
        """
        if self.training:
            # Check if mu is 3D (separate for each encoder)
            if len(mu.shape) == 3:
                batch_size, latent_dim, num_encoders = mu.shape
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            else:  # 2D case (original)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
        return mu
    
    def forward(self, x_list, masks):
        """
        Optimized forward pass with vectorized processing

        Returns:
            nn_output: NN decoder outputs [batch_size, nn_output_dim, num_encoders]
            fmri_outputs: List of fMRI decoder outputs per subject
            mu: Base Gaussian means [batch_size, latent_dim, num_encoders]
            logvar: Base Gaussian log variances [batch_size, latent_dim, num_encoders]
        """
        batch_size = masks.size(0)
        device = masks.device
        encoders_to_use = self.encoders
        num_encoders = len(self.encoders)
        latent_dim = self.encoders[0].latent_dim

        # Determine which encoders and inputs to use
        if self.only_nn_encoder:
            # For inputs, use the NN data (either the only input or the last one)
            inputs_to_use = [x_list[0] if len(x_list) == 1 else x_list[-1]]
            masks_to_use = masks[:, 0].unsqueeze(1) if masks.size(1) == 1 else masks[:, -1].unsqueeze(1)
        else:
            inputs_to_use = x_list
            masks_to_use = masks
        
        # Pre-allocate tensors to store latent representations
        mu = torch.zeros(batch_size, latent_dim, num_encoders, device=device)
        logvar = torch.zeros(batch_size, latent_dim, num_encoders, device=device)
        valid_encoder_samples = torch.zeros(batch_size, num_encoders, dtype=torch.bool, device=device)
        
        # Pre-allocate tensor for the latent codes
        z = torch.zeros(batch_size, latent_dim, num_encoders, device=device)

        # Group encoders by architecture for batch processing
        fmri_encoder_indices = list(range(self.num_fmri_encoders))
        nn_encoder_indices = [self.nn_encoder_idx] if self.nn_encoder_idx >= 0 else []
        
        # Process fMRI encoders in batches where possible
        if fmri_encoder_indices:
            # Find samples with same valid encoders for batch processing
            valid_mask_patterns = {}
            for i in range(batch_size):
                pattern = tuple(masks_to_use[i, idx].item() for idx in fmri_encoder_indices)
                if pattern not in valid_mask_patterns:
                    valid_mask_patterns[pattern] = []
                valid_mask_patterns[pattern].append(i)
            
            # Process each group of samples with same pattern
            for pattern, sample_indices in valid_mask_patterns.items():
                sample_indices = torch.tensor(sample_indices, device=device)
                
                # Skip if all encoders are invalid for these samples
                if not any(pattern):
                    continue
                    
                # Gather all valid encoder inputs for these samples
                for e_idx, is_valid in enumerate(pattern):
                    if is_valid:
                        orig_idx = fmri_encoder_indices[e_idx]
                        valid_encoder_samples[sample_indices, orig_idx] = True
                        
                        # Get encoder input for these samples
                        encoder_input = inputs_to_use[orig_idx][sample_indices]
                        
                        # Encode to base Gaussian parameters
                        encoder_mu, encoder_logvar = self.encoders[orig_idx](encoder_input)
                        
                        # Store base Gaussian parameters
                        mu[sample_indices, :, orig_idx] = encoder_mu
                        logvar[sample_indices, :, orig_idx] = encoder_logvar
                        
                        # Sample from base Gaussian using reparameterization
                        z_0 = self.reparameterize(encoder_mu, encoder_logvar)

                        # Store sampled latent code
                        z[sample_indices, :, orig_idx] = z_0
        
        # Process NN encoder separately (if present and active)
        if nn_encoder_indices and not self.only_fmri_encoders:
            nn_idx = nn_encoder_indices[0]
            nn_mask = masks_to_use[:, nn_idx]
            if nn_mask.sum() > 0:
                nn_valid_indices = torch.where(nn_mask)[0]
                valid_encoder_samples[nn_valid_indices, nn_idx] = True
                
                # Get NN encoder input
                nn_input = inputs_to_use[nn_idx][nn_valid_indices]
                
                # Encode to base Gaussian parameters
                nn_mu, nn_logvar = self.encoders[nn_idx](nn_input)
                
                # Store base Gaussian parameters
                mu[nn_valid_indices, :, nn_idx] = nn_mu
                logvar[nn_valid_indices, :, nn_idx] = nn_logvar
                
                # Sample from base Gaussian using reparameterization
                z_0 = self.reparameterize(nn_mu, nn_logvar)

                # Store sampled latent code
                z[nn_valid_indices, :, nn_idx] = z_0
        
        # Handle decoders more efficiently
        nn_output = None
        fmri_outputs = None
        
        # NN decoder processing
        if self.use_nn_decoder and self.nn_decoder is not None:
            nn_output_dim = self.nn_decoder.output_dim
            nn_output = torch.zeros(batch_size, nn_output_dim, num_encoders, device=device)
            
            # Process all encoders in a vectorized way if possible
            for i in range(num_encoders):
                indices = torch.where(valid_encoder_samples[:, i])[0]
                if len(indices) > 0:
                    decoder_output = self.nn_decoder(z[indices, :, i])
                    nn_output[indices, :, i] = decoder_output
        
        # fMRI decoders processing
        if self.use_fmri_decoders and self.fmri_decoders is not None:
            num_subjects = len(self.fmri_decoders)
            fmri_outputs = []
            
            for subject_idx in range(num_subjects):
                subject_decoder = self.fmri_decoders[subject_idx]
                output_dim = subject_decoder.output_dim
                subject_output = torch.zeros(batch_size, output_dim, num_encoders, device=device)
                
                for i in range(num_encoders):
                    indices = torch.where(valid_encoder_samples[:, i])[0]
                    if len(indices) > 0:
                        decoder_output = subject_decoder(z[indices, :, i])
                        subject_output[indices, :, i] = decoder_output
                
                fmri_outputs.append(subject_output)
        
        return nn_output, fmri_outputs, mu, logvar

       