// Add your Prisma database configuration code here
import { PrismaClient as PrismaClientExternalRE } from '@prisma/client-external-re';


/**
 * External Real Estate Database Client
 *
 * READ-ONLY connection to external property listings database
 *
 * Features:
 * - Property listings from MLS/RE sources
 * - Market intelligence (county & national metrics)
 * - Query logging in development
 * - Connection pooling
 *
 * IMPORTANT: This is READ-ONLY. Do not attempt writes.
 */


const globalForPrismaRE = globalThis as unknown as {
  prismaRE: PrismaClientExternalRE | undefined;
};


function createExternalREClient() {
  return new PrismaClientExternalRE({
    log: process.env.NODE_ENV === 'development'
      ? ['query', 'error', 'warn']
      : ['error'],
  });
}


export const prismaRE =
  globalForPrismaRE.prismaRE ?? createExternalREClient();


if (process.env.NODE_ENV !== 'production') {
  globalForPrismaRE.prismaRE = prismaRE;
}


/**
 * Check external RE database connection
 */
export async function checkExternalREConnection(): Promise<boolean> {
  try {
    await prismaRE.$queryRaw`SELECT 1`;
    return true;
  } catch (error) {
    console.error('External RE DB connection failed:', error);
    return false;
  }
}


/**
 * Get database statistics
 */
export async function getExternalREStats() {
  try {
    const [listings, counties, nationalMetrics] = await Promise.all([
      prismaRE.property_listings.count(),
      prismaRE.county_economic_metrics.count(),
      prismaRE.national_economic_metrics.count(),
    ]);


    return {
      totalListings: listings,
      totalCounties: counties,
      totalNationalMetrics: nationalMetrics,
      connected: true,
    };
  } catch (error) {
    console.error('Failed to get external RE stats:', error);
    return {
      totalListings: 0,
      totalCounties: 0,
      totalNationalMetrics: 0,
      connected: false,
    };
  }
}
