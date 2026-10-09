weather = input("Weather (warm/cool): ").lower()
occasion = input("Occasion (casual/formal): ").lower()
vibe = input("Vibe (casual/elegant): ").lower()


if weather == "warm" and occasion == "casual":
    top = "Cotton T-shirt"
    bottom = "Blue jeans"
    shoes = "Sneakers"

elif weather == "cool" and occasion == "casual":
    top = "Sweater"
    bottom = "Black jeans"
    shoes = "Sneakers"

elif occasion == "formal":
    top = "Formal shirt"
    bottom = "Black trousers"
    shoes = "Formal shoes"

else:
    top = "Basic top"
    bottom = "Jeans"
    shoes = "Sneakers"


# Modify outfit according to vibe
if vibe == "elegant":
    shoes = "Loafers"
    
if vibe == "casual":
    shoes = "Sneakers"


print("\n👗 Recommended Outfit")
print("Top:", top)
print("Bottom:", bottom)
print("Shoes:", shoes)