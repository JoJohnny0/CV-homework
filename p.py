layers = [
    (3, 2, 1, 0),  # ConvTranspose2d(in_channels, in_channels*2, 3, stride=2, padding=1)
    (9, 3, 1, 0),  # ConvTranspose2d(in_channels*2, in_channels*4, 9, stride=3, padding=1)
    (7, 5, 1, 0),  # ConvTranspose2d(in_channels*4, in_channels*4, 7, stride=5, padding=1)
    (9, 2, 0, 0),  # ConvTranspose2d(in_channels*4, in_channels*2, 9, stride=2)
    (6, 1, 0, 0),  # ConvTranspose2d(in_channels*2, in_channels, 6, stride=1)
    (11, 1, 0, 0)   # ConvTranspose2d(in_channels, 3, 11, stride=1)
]

def get_dims(output_size, kernel_size, stride, default_padding = 0, default_output_padding = 0):

    default_input_size: float = 1 + ((output_size - kernel_size - default_output_padding + 2 * default_padding) / stride)
    padding: int
    out_padding: int
    input_size: int
    if default_input_size.is_integer():
        out_padding = default_output_padding
        padding = default_padding
        input_size = int(default_input_size)
    else:
        size_mismatch: int = (kernel_size + stride - output_size) % stride
        out_padding = size_mismatch % 2
        padding = (size_mismatch + out_padding) // 2
        input_size = 1 + ((output_size - kernel_size - out_padding + 2 * padding) // stride)

    if input_size <= 0:
        padding += (stride * (2 - input_size)) // 2
        input_size = 1

    return input_size, padding, out_padding

output_size = 256
decoder_layers = []
for layer in reversed(layers):
    kernel_size, stride, default_padding, default_output_padding = layer
    input_size, padding, out_padding = get_dims(output_size, kernel_size, stride, default_padding, default_output_padding)
    
    output_size = input_size  # Update output size for the next layer

