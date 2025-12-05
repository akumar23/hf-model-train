#!/usr/bin/env python3
"""
Example script demonstrating how to use a trained model for inference.

Usage:
    python inference_example.py --model path/to/model --prompt "Your prompt here"
"""

import argparse
from utils import load_model_for_inference, generate_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained PEFT model"
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained PEFT model directory'
    )
    parser.add_argument(
        '--prompt',
        required=True,
        help='Input prompt for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Maximum tokens to generate (default: 200)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Nucleus sampling top-p (default: 0.9)'
    )
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=1,
        help='Number of sequences to generate (default: 1)'
    )
    parser.add_argument(
        '--no-sample',
        action='store_true',
        help='Disable sampling (use greedy decoding)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PEFT Model Inference")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"\nGeneration settings:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Sampling: {'disabled' if args.no_sample else 'enabled'}")
    print("\n" + "=" * 80)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_for_inference(args.model)

    # Generate text
    print("\nGenerating...\n")
    generated_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        num_return_sequences=args.num_sequences
    )

    # Print results
    print("=" * 80)
    print("Generated Text")
    print("=" * 80)

    for idx, text in enumerate(generated_texts, 1):
        if args.num_sequences > 1:
            print(f"\n[Sequence {idx}]")
        print(text)
        if idx < len(generated_texts):
            print("\n" + "-" * 80)

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
