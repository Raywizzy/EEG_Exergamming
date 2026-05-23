#!/bin/bash
# Quick badge refresh command
# Usage: ./scripts/refresh_badges.sh

echo "🔄 Refreshing validation status badges..."

# Generate new badges
python scripts/create_current_badges.py

# Update README
python scripts/update_validation_status.py

echo "✅ Badge refresh complete!"
echo ""
echo "📊 Your current validation status:"
echo "• Overall BA: 65.5% (PASS Clinical Threshold)"
echo "• ds004584: 60.3% BA (117 subjects)"
echo "• ds002778: 70.8% BA (10 subjects)"
echo ""
echo "Badge files updated in results/badges/"