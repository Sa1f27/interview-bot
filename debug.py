#!/usr/bin/env python3
"""
Test the currently running server at 192.168.48.18:8060
"""

import requests
import json

def test_server():
    """Test various endpoints on the running server"""
    base_url = "http://192.168.48.18:8060"
    
    endpoints_to_test = [
        "/",
        "/healthz", 
        "/test",
        "/debug/mounted-apps",
        "/daily_standup/",
        "/daily_standup/health",
        "/daily_standup/test", 
        "/daily_standup/start_test"
    ]
    
    print(f"🔍 Testing server at {base_url}")
    print("=" * 50)
    
    for endpoint in endpoints_to_test:
        url = f"{base_url}{endpoint}"
        print(f"\n📡 Testing: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                print("  ✅ SUCCESS")
                try:
                    data = response.json()
                    print(f"  📄 Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"  📄 Response: {response.text[:200]}...")
            elif response.status_code == 404:
                print("  ❌ NOT FOUND")
                print(f"  📄 Error: {response.text}")
            elif response.status_code == 422:
                print("  ⚠️ VALIDATION ERROR")
                print(f"  📄 Error: {response.text}")
            else:
                print(f"  ⚠️ Status {response.status_code}")
                print(f"  📄 Response: {response.text[:200]}...")
                
        except requests.exceptions.ConnectionError:
            print("  ❌ CONNECTION REFUSED - Server not running")
        except requests.exceptions.Timeout:
            print("  ❌ TIMEOUT - Server not responding")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

if __name__ == "__main__":
    test_server()