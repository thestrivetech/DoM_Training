// Add your database test code here
import { prismaRE, checkExternalREConnection, getExternalREStats } from './src/lib/database/prisma';


async function testRealEstateDatabase() {
  console.log('🔍 Testing Real Estate Database Connection...\n');


  try {
    // Step 1: Check basic connection
    console.log('1️⃣ Checking connection...');
    const isConnected = await checkExternalREConnection();


    if (!isConnected) {
      console.error('❌ Failed to connect to database');
      console.log('\n📝 Troubleshooting:');
      console.log('   - Check .env.local has REAL_ESTATE_DATABASE_URL');
      console.log('   - Verify credentials are correct');
      console.log('   - Make sure you have internet connection');
      process.exit(1);
    }


    console.log('✅ Connected successfully!\n');


    // Step 2: Get database statistics
    console.log('2️⃣ Fetching database stats...');
    const stats = await getExternalREStats();


    console.log('📊 Database Statistics:');
    console.log(`   Total Listings:        ${stats.totalListings.toLocaleString()}`);
    console.log(`   County Metrics:        ${stats.totalCounties.toLocaleString()}`);
    console.log(`   National Indicators:   ${stats.totalNationalMetrics.toLocaleString()}`);
    console.log(`   Connection Status:     ${stats.connected ? '✅ Connected' : '❌ Disconnected'}`);
    console.log('');


    // Step 3: Get a sample listing
    console.log('3️⃣ Fetching sample listing...');
    const sampleListing = await prismaRE.property_listings.findFirst({
      where: { status: 'Active' },
    });


    if (!sampleListing) {
      console.log('⚠️  No active listings found in database');
    } else {
      console.log('🏠 Sample Listing:');
      console.log(`   ID:       ${sampleListing.id}`);
      console.log(`   Address:  ${sampleListing.formattedAddress}`);
      console.log(`   City:     ${sampleListing.city}, ${sampleListing.state} ${sampleListing.zipCode || ''}`);
      console.log(`   Price:    $${Number(sampleListing.price || 0).toLocaleString()}`);
      console.log(`   Bedrooms: ${sampleListing.bedrooms || 'N/A'}`);
      console.log(`   Bathrooms: ${sampleListing.bathrooms ? Number(sampleListing.bathrooms) : 'N/A'}`);
      console.log(`   Sqft:     ${sampleListing.squareFootage?.toLocaleString() || 'N/A'}`);
      console.log(`   Type:     ${sampleListing.propertyType || 'N/A'}`);
      console.log(`   Status:   ${sampleListing.status || 'N/A'}`);
    }


    console.log('\n✅ All tests passed! You\'re ready to start querying data.\n');


    process.exit(0);
  } catch (error) {
    console.error('\n❌ Test failed with error:');
    console.error(error);


    console.log('\n📝 Common issues:');
    console.log('   - Database credentials incorrect');
    console.log('   - Network connectivity problems');
    console.log('   - Prisma client not generated (run: npx prisma generate --schema=prisma/schema-external-re.prisma)');


    process.exit(1);
  }
}


testRealEstateDatabase();