#! /bin/bash
prompt=<<EOF
EOF
curl --location 'https://smi-api.sponge-theory.dev/v1/text-to-image' \
--header 'x-smi-key: s78d8z6sdx-d058-4dd4-9c93-24761122aec5' \
--header 'accept: image/png' \
--header 'Content-Type: application/json' \
--output generated_image.png \
--data @- << 'EOF'
{
    "text_prompts": [
        {
            "text": "A luxurious, futuristic hotel innovation center or high-tech operations hub, designed with polished marble floors, golden ambient lighting, sleek metallic surfaces, and advanced digital interfaces, exuding refinement and exclusivity.At the center, a massive holographic architectural blueprint of an AI-powered technical infrastructure floats above a sleek, illuminated data console. The visualization should showcase interconnected data networks, AI processing layers, cloud storage hubs, and security protocols, seamlessly integrated into the refined luxury space.Elegant glowing data streams flow between interactive panels and intelligent servers, symbolizing the AI’s structured workflow and processing power. Subtle architectural design elements, such as glass walls with embedded digital schematics, reinforce the concept of precision, control, and efficiency in AI-driven hotel operations.The environment remains minimalist, immaculate, and intelligent, merging state-of-the-art technology with the sophistication of luxury hospitality. No people—just an exquisite, high-tech environment where AI-driven architecture is elegantly displayed as the foundation of future innovation.",
            "weight": 1
        }
    ],
    "width": 512,
    "height": 768,
    "seed": 45851257,
    "worker_id": "fluxDev",
    "steps": 15
}
EOF