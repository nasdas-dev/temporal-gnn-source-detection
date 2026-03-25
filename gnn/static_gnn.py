import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv, GINConv


class StaticGNN(torch.nn.Module):
    def __init__(self, num_preprocess_layers, embed_dim_preprocess, num_postprocess_layers, num_conv_layers, aggr,
                 num_node_features, hidden_channels, num_classes, dropout_rate, batch_norm, skip):
        super(StaticGNN, self).__init__()

        # Save number of preprocessing and postprocessing layers in self.
        self.num_preprocess_layers = num_preprocess_layers
        self.num_postprocess_layers = num_postprocess_layers
        # Save embedding dimensionality of preprocessing layers.
        self.embed_dim_preprocess = embed_dim_preprocess

        # Save num_conv_layers in self.
        self.num_conv_layers = num_conv_layers
        # Save num_classes in self.
        self.num_classes = num_classes
        # Save dropout rate in self.
        self.dropout_rate = dropout_rate
        # Save boolean wrt batch normalization in self.
        self.batch_norm = batch_norm
        # Save boolean wrt skip connections in self.
        self.skip = skip

        # Initialize lists for preprocessing, convolution, postprocessing, and final layers.
        self.preprocess = torch.nn.Sequential()
        self.convs = torch.nn.ModuleList()
        self.banor = torch.nn.ModuleList()  # Batch normalization for graph conv. layers.
        self.activ = torch.nn.ModuleList()  # Learnable (PReLU) activations for graph conv. layers.
        self.postprocess = torch.nn.Sequential()
        self.final = torch.nn.Sequential()  # Final linear layer (which, optionally, skip connection points to).

        # -----------------------------------------------------------------------
        # PREPROCESSING LAYERS

        # Setup preprocessing if number of preproc. layers is larger than 0.
        if self.num_preprocess_layers > 0:
            # Add layers sequentially.
            for l in range(self.num_preprocess_layers):
                # Input size is different for first layer.
                in_features = num_node_features if l == 0 else self.embed_dim_preprocess
                # Linear layer
                self.preprocess.append(Linear(in_features, self.embed_dim_preprocess))
                # Optional batch normalization
                if self.batch_norm:
                    self.preprocess.append(torch.nn.BatchNorm1d(self.embed_dim_preprocess))
                # PReLU activation
                self.preprocess.append(torch.nn.PReLU())
                # Dropout
                self.preprocess.append(torch.nn.Dropout(p=self.dropout_rate))

        # -----------------------------------------------------------------------
        # CONVOLUTION LAYERS

        # Create layers based on num_conv_layers.
        for l in range(self.num_conv_layers):
            # Input size is different for first layer (and this also depends on whether preprocessing is done or not).
            in_channels = num_node_features if (
                        self.num_preprocess_layers == 0 and l == 0) else self.embed_dim_preprocess if (
                        self.num_preprocess_layers > 0 and l == 0) else hidden_channels
            # Graph convolution layer
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr=aggr))
            # Optional batch normalization
            if self.batch_norm:
                self.banor.append(torch.nn.BatchNorm1d(hidden_channels))
            # PReLU activation
            self.activ.append(torch.nn.PReLU())

        # -----------------------------------------------------------------------
        # POSTPROCESSING LAYERS

        # Setup postprocessing if number of postproc. layers is larger than 0.
        # Here, we do not change the dimensionality of the embeddings.
        # That is why in- and out-features are simply 'hidden_channels'.
        if self.num_postprocess_layers > 0:
            # Add layers sequentially.
            for _ in range(self.num_postprocess_layers):
                # Linear layer
                self.postprocess.append(Linear(hidden_channels, hidden_channels))
                # Optional batch normalization
                if self.batch_norm:
                    self.postprocess.append(torch.nn.BatchNorm1d(hidden_channels))
                # PReLU activation
                self.postprocess.append(torch.nn.PReLU())
                # Dropout
                self.postprocess.append(torch.nn.Dropout(p=self.dropout_rate))

        # -----------------------------------------------------------------------
        # FINAL LAYER

        # Determine number of input channels (depending on skip).
        in_channels = (hidden_channels + num_node_features) if self.skip else hidden_channels
        # Add final linear layer. Output channels must be 1.
        self.final.append(Linear(in_channels, 1))
        # Optional batch normalization
        if self.batch_norm:
            self.final.append(torch.nn.BatchNorm1d(1))
        # PReLU activation
        self.final.append(torch.nn.PReLU())

    def forward(self, x, edge_index, edge_weights, batch):

        # Preprocessing layers
        x1 = self.preprocess(x)

        # Iterate over graph convolution layers.
        # Note: this is different from preprocessing layers since we also
        # need to pass edge_index here. Thus, we make this a bit more explicite.
        for i, conv in enumerate(self.convs):
            # Graph convolution
            x1 = conv(x1, edge_index, edge_weights)
            # Batch normalization (if it is "on").
            if self.batch_norm:
                x1 = self.banor[i](x1)
            # Activation
            x1 = self.activ[i](x1)
            # Dropout
            x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)

        # Linear layer
        # The x that the linear layer receives as input has
        # dimensions (batch_size * n_nodes, hidden_channels).
        # This tensor is then multiplied with a linear layer
        # that computes a weighted sum of the 32 embeddings for
        # each node separately. What results is a tensor of
        # dimensions (batch_size * n_nodes, 1). We have to
        # reshape this so we get a tensor of dimensions
        # (batch_size, n_nodes). The following log_softmax
        # function then normalizes row-wise.
        x1 = self.postprocess(x1)
        # x = self.lin(x).reshape((batch.unique().shape[0], self.num_classes))

        # Final layer with skip connection.
        if self.skip:
            x1 = self.final(torch.cat([x, x1], dim=-1)).reshape((batch.unique().shape[0], self.num_classes))
        # Final layer without skip connection.
        else:
            x1 = self.final(x1).reshape((batch.unique().shape[0], self.num_classes))

        # Log-softmax
        x1 = F.log_softmax(x1, dim=1)

        return x1