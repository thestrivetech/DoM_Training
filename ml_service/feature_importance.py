#!/usr/bin/env python3
import sys
import json
import joblib
from pathlib import Path

def main():
    try:
        model_dir = Path(__file__).parent / 'models'

        if not model_dir.exists():
            print(json.dumps({'error': 'Model not found. Please train the model first.'}), file=sys.stderr)
            sys.exit(1)

        # Load model and feature names
        model = joblib.load(model_dir / 'model.pkl')
        feature_names = joblib.load(model_dir / 'feature_names.pkl')

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = [
                {
                    'feature': feat,
                    'importance': float(imp),
                    'importance_percentage': float(imp / sum(importances) * 100)
                }
                for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            ]

            result = {
                'feature_importance': feature_importance,
                'total_features': len(feature_names),
                'top_features': feature_importance[:10]
            }

            print(json.dumps(result))
            sys.exit(0)
        else:
            print(json.dumps({'error': 'Model does not support feature importance'}), file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
