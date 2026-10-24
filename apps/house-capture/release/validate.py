#!/usr/bin/env python3
"""Read-only local submission checks; does not replace Organizer/App Store validation."""
import json
import plistlib
import sys
from pathlib import Path
from urllib.parse import urlparse

root = Path(__file__).resolve().parent
app = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent / 'HouseCapture'
errors = []
info = plistlib.loads((app / 'Info.plist').read_bytes())
privacy_file = app / 'PrivacyInfo.xcprivacy'
if not privacy_file.exists():
    errors.append('PrivacyInfo.xcprivacy missing from app bundle')
else:
    privacy = plistlib.loads(privacy_file.read_bytes())
    reasons = {item['NSPrivacyAccessedAPIType']: item['NSPrivacyAccessedAPITypeReasons'] for item in privacy.get('NSPrivacyAccessedAPITypes', [])}
    for category, reason in [('FileTimestamp', 'C617.1'), ('DiskSpace', 'E174.1')]:
        if reason not in reasons.get('NSPrivacyAccessedAPICategory' + category, []):
            errors.append(f'Missing required reason for {category}')
    if privacy.get('NSPrivacyTracking') or privacy.get('NSPrivacyCollectedDataTypes'):
        errors.append('Re-audit store privacy answers: tracking or data collection declared')
# Make photoreal uploads to a user-configured server; only local-network HTTP is exempted.
ats = info.get('NSAppTransportSecurity', {})
if set(ats) - {'NSAllowsLocalNetworking'}:
    errors.append('Review App Transport Security exceptions: ' + ', '.join(sorted(set(ats) - {'NSAllowsLocalNetworking'})))
if ats.get('NSAllowsLocalNetworking') and not info.get('NSLocalNetworkUsageDescription'):
    errors.append('Local networking needs NSLocalNetworkUsageDescription')
for resource in ['Support.json', 'PrivacyPolicy.json']:
    if not (app / resource).exists(): errors.append(f'Missing offline help resource: {resource}')
for key in ['HouseCapturePrivacyURL', 'HouseCaptureSupportURL']:
    url = urlparse(info.get(key, ''))
    if url.scheme != 'https' or not url.netloc:
        errors.append(f'Public HTTPS URL still required: {key}')
metadata = json.loads((root / 'app-store-metadata.json').read_text())
for key, limit in [('name', 30), ('subtitle', 30), ('keywords', 100), ('description', 4000)]:
    if len(metadata.get(key, '')) > limit: errors.append(f'{key} exceeds its {limit}-character limit')
for key in ['supportURL', 'privacyPolicyURL', 'copyright', 'price', 'reviewContact']:
    if metadata.get(key) is None: errors.append(f'App Store metadata still required: {key}')
if info.get('ITSAppUsesNonExemptEncryption') is not False:
    errors.append('Review export-compliance declaration')
if not info.get('NSCameraUsageDescription'):
    errors.append('Missing camera purpose string')
for name in ['iphone-6.9-library.png', 'ipad-13-library.png']:
    if not (root / 'screenshots' / name).exists(): errors.append('Missing reviewed screenshot: ' + name)
for error in errors: print('TODO:', error)
print('Local checks passed.' if not errors else f'{len(errors)} release prerequisites remain. See release-checklist.txt for physical-device and account checks.')
sys.exit(bool(errors))
