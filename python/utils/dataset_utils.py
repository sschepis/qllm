import tensorflow as tf

def create_dataset(data, batch_size=32, input_dtype=tf.int32, target_dtype=tf.int32):
    """
    Creates a TensorFlow dataset from a list of input/target pairs.

    Args:
        data (list): A list where each element is a tuple or list [input_data, target_data].
                     input_data and target_data should be sequences (like lists or numpy arrays)
                     of numbers (e.g., token IDs).
        batch_size (int): The batch size for the dataset.
        input_dtype (tf.DType): TensorFlow dtype for input tensors. Defaults to tf.int32.
        target_dtype (tf.DType): TensorFlow dtype for target tensors. Defaults to tf.int32.

    Returns:
        tf.data.Dataset: A batched TensorFlow dataset yielding tuples (inputs, targets).
    """

    def generator():
        for input_item, target_item in data:
            # Yield input and target tensors directly
            # Keras expects a tuple (inputs, targets) or (inputs, targets, sample_weights)
            yield tf.constant(input_item, dtype=input_dtype), tf.constant(target_item, dtype=target_dtype)

    # Determine the output shapes based on the first item, assuming uniform sequence lengths
    # If sequence lengths vary, output_shapes should be set to (None,) or padded/bucketed.
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=input_dtype), # Assuming variable length sequences
        tf.TensorSpec(shape=(None,), dtype=target_dtype)
    )
    # If sequence lengths are fixed, you could use:
    # first_input_len = len(data[0][0]) if data else 0
    # first_target_len = len(data[0][1]) if data else 0
    # output_signature = (
    #     tf.TensorSpec(shape=(first_input_len,), dtype=input_dtype),
    #     tf.TensorSpec(shape=(first_target_len,), dtype=target_dtype)
    # )


    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # Batch the dataset
    if batch_size > 0:
        dataset = dataset.batch(batch_size)

    # Optional: Add prefetching for performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

# Example Usage (Illustrative)
# if __name__ == '__main__':
#     # Sample data (list of [input_sequence, target_sequence])
#     sample_data = [
#         ([1, 2, 3, 4], [2, 3, 4, 5]),
#         ([10, 11, 12], [11, 12, 13]),
#         ([5, 6, 7, 8, 9], [6, 7, 8, 9, 10])
#     ]
#     batch_sz = 2
#
#     tf_dataset = create_dataset(sample_data, batch_size=batch_sz)
#
#     print(f"Dataset created with batch size {batch_sz}")
#     for batch_inputs, batch_targets in tf_dataset.take(2): # Take a couple of batches
#         print("-" * 20)
#         print("Batch Inputs Shape:", batch_inputs.shape)
#         print("Batch Inputs:", batch_inputs.numpy())
#         print("Batch Targets Shape:", batch_targets.shape)
#         print("Batch Targets:", batch_targets.numpy())